# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agnostic model initialization with huggingface
"""

import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import AdvantageEstimator, RayPPOTrainer, _timer, apply_kl_penalty, compute_advantage

import os
import ray
import subprocess
import shutil

import yaml
from pathlib import Path
from verl.utils.fewshots.few_shot_manager import FewShotConfig, FewShotManager
from verl.utils.model import compute_position_id_with_mask
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance

"""
Batch structure for final loss computation:
batch:
    .batch:
        - advantages: b * response_len
        - attention_mask: b * all_len
        - entroys: b * response_len
        - input_ids: b * all_len
        - old_log_probs: b * response_len
        - position_ids: b * all_len
        - ref_log_prob: b * response_len
        - response_mask: b * response_len
        - responses: b * response_len
        - returns: b * response_len
        - token_level_rewards: b * response_len
        - token_level_scores: b * response_len
    .non_tensor_batch:
        - data_source: array, b
        - ability
        - index
        - uid
        - acc
        - __index_level_0__
        - reward_model: array of dict, b
    .meta_info:
        - global_token_sum: list, b. actual response_len + prompt_len
        - temperature

Following verl's API, we construct 4 objects: new_train_batch, new_infer_batch, train_batch, infer_batch
"""


class RayDAPOFewshotTrainer(RayPPOTrainer):
    """
    DAPO Trainer enhanced with Few-Shot Learning capabilities.
    
    Key Features:
    - Supports multiple rollouts with different few-shot examples
    - Maintains one baseline rollout without few-shot (for comparison)
    - Random selection strategy
    - Minimal modification to the original trainer logic
    """

    def __init__(self, config, *args, **kwargs):
        """
        Initialize the trainer with few-shot enhancement.
        """

        super().__init__(config, *args, **kwargs)

        # Initialize Few-Shot Manager
        few_shot_config_path = getattr(config.algorithm, 'few_shot_config_path', None)
        if few_shot_config_path is not None:
            config_path = Path(few_shot_config_path)
            with open(config_path, 'r') as f:
                few_shot_config_dict = yaml.safe_load(f)
            few_shot_config_dict = self._override_config(
                few_shot_config_dict, 
                config.algorithm.few_shot_config
            )
            few_shot_config = FewShotConfig(**few_shot_config_dict)
        else:
            raise FileNotFoundError(f"Few-shot config file not found!")

        self.few_shot_manager = FewShotManager(few_shot_config)

        # Flag to track if few-shot is enabled
        self.use_few_shot = few_shot_config.enabled and len(self.few_shot_manager.examples) > 0

        if self.use_few_shot:
        
            self.rollouts_with_fewshots = min(few_shot_config.num_groups, self.config.actor_rollout_ref.rollout.n)
            self.rollouts_without_fewshots = self.config.actor_rollout_ref.rollout.n - self.rollouts_with_fewshots

            if self.rollouts_with_fewshots == 0:
                self.use_few_shot = False

            else:
                print(f"Few-Shot Enhancement Enabled:")
                print(f"  - Strategy: {few_shot_config.selection_strategy}")
                print(f"  - Shots for each rollout: {few_shot_config.num_fewshots}")
                print(f"  - Total examples: {len(self.few_shot_manager.examples)}")
                print(f"  - Rollouts with few-shot: {self.rollouts_with_fewshots}")
                print(f"  - Baseline rollouts: {self.rollouts_without_fewshots}")
        
        if not self.use_few_shot:
            print(f"Few-Shot Enhancement NOT Enabled.")

    def _override_config(self, base_config, cli_config):
        for field in dir(cli_config):
            if not field.startswith("_"):
                if hasattr(cli_config, field):
                    base_config[field] = cli_config[field]

        return base_config 

    def _prepare_batch_dict_with_fewshot(self, batch_dict):
        """
        Prepare generation batch with few-shot examples if enabled.
        
        This method handles the core logic of augmenting prompts with few-shot examples.
        If few-shot is disabled, returns the original batch.
        """

        if not self.use_few_shot:
            return batch_dict
        
        # Augment batch with few-shot examples
        augmented_batch_dict = self.few_shot_manager.augment_batch_dict(
            batch_dict=batch_dict,
            num_rollouts_with_fewshots=self.rollouts_with_fewshots,
            num_rollouts_without_fewshots=self.rollouts_without_fewshots,
            max_prompt_length=self.config.data.max_prompt_length,
            truncation=self.config.data.truncation,
            tokenizer=self.tokenizer,
            template_format=self.config.algorithm.few_shot_config.template_format
        )

        return augmented_batch_dict

    def _union_gen_batch_output(self, new_batch, gen_batch_output):
        """
        Union the generation output with the batch.
        gen_batch_output only has .batch: attention_mask, input_ids, position_ids, prompts, responses
        """
        prompt_len = new_batch.batch["attention_mask"].size(1)
        new_batch.batch["prompts"] = new_batch.batch["input_ids"]
        new_batch.batch["responses"] = gen_batch_output.batch["responses"]
        new_batch.batch["input_ids"] = torch.cat([
            new_batch.batch["input_ids"],           # shape: (batch, prompt_len)
            gen_batch_output.batch["responses"]     # shape: (batch, response_len)
        ], dim=-1)
        new_batch.batch["attention_mask"] = torch.cat([
            new_batch.batch["attention_mask"], 
            gen_batch_output.batch["attention_mask"][:, prompt_len:], 
        ], dim=-1)
        new_batch.batch["position_ids"] = compute_position_id_with_mask(
            new_batch.batch["attention_mask"]
        )
        
        return new_batch

    def _balance_batch(self, train_batch: DataProto, infer_batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = train_batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = train_batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        train_batch.reorder(global_idx)
        infer_batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)
        
        
    def fit(self):
        """
        The training loop of PPO with Few-Shot Enhancement.
        
        Main modifications from the original:
        1. Augment prompts with few-shot examples before generation
        2. Handle expanded batch size for multiple rollouts
        3. Update few-shot selection statistics after reward computation
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            self._report_timing_stat()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        self._report_timing_stat()

        # Add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        timing_raw = defaultdict(float)
        train_batch = None
        infer_batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0

        RAY_JOB_SUBMISSION_ID = ray.runtime_context.get_runtime_context().get_job_id()
        run_path = None
        print(f"RAY_JOB_SUBMISSION_ID:{RAY_JOB_SUBMISSION_ID}")

        assert self.config.algorithm.train_infer_mode in [
            "augmented+augmented",
            "augmented+original",
            "original+augmented",
            "original+original"
        ], f"{self.config.algorithm.train_infer_mode} not supported!"
        train_infer_mode = self.config.algorithm.train_infer_mode.split("+")
        train_mode = train_infer_mode[0]
        infer_mode = train_infer_mode[-1]

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}

                # Prepare the augmented batch with few-shot examples
                augmented_batch_dict = self._prepare_batch_dict_with_fewshot(batch_dict)

                # Convert both batch_dict and augmented_batch_dict to DataProto
                original_new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                augmented_new_batch: DataProto = DataProto.from_single_dict(augmented_batch_dict)

                num_gen_batches += 1

                # Pop keys for generation
                # Generation uses augmented_new_batch; note that rollout count is effectively 1 here
                if "multi_modal_inputs" in augmented_new_batch.non_tensor_batch.keys():
                    gen_batch = augmented_new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "multi_modal_inputs"],
                    )
                else:
                    gen_batch = augmented_new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )

                # Each prompt generates only one response
                if self.use_few_shot:
                    gen_batch.meta_info["n"] = 1

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # Generate responses
                    with _timer("gen", timing_raw):
                        # gen_batch_output contains:
                        # .batch: attention_mask(b*total_len), input_ids(b*total_len), 
                        # position_ids(b*total_len), prompts(b*input_len), responses(b*response_len)
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            augmented_new_batch = augmented_new_batch.union(gen_baseline_output)

                            if not self.use_rm_worker:
                                reward_baseline_tensor = self.reward_fn(augmented_new_batch)
                            else:
                                new_batch_padded, pad_size = pad_dataproto_to_divisor(augmented_new_batch, self.rm_worker_wg.world_size)
                                reward_baseline_tensor = self.rm_worker_wg.compute_reward(new_batch_padded)
                                reward_baseline_tensor = unpad_dataproto(reward_baseline_tensor, pad_size)
                                reward_baseline_tensor = reward_baseline_tensor.batch['reward_tensor']
                                self._report_timing_stat()

                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)
                            augmented_new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                            augmented_new_batch.batch["reward_baselines"] = reward_baseline_tensor
                            del gen_baseline_batch, gen_baseline_output
                    
                    # Generate unique identifiers on the original batch, then distribute to the augmented batch
                    original_new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(original_new_batch.batch))], dtype=object
                    )
                    original_new_batch = original_new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    augmented_new_batch.non_tensor_batch["uid"] = original_new_batch.non_tensor_batch["uid"].copy()

                    # Union operation: add generated content to the batch
                    # We need to do union on both new_batch and augmented_new_batch,
                    # as one may be used for ref_probs and the other for actor_probs
                    # Union requires matching keys to have the same values,
                    # so for original_new_batch we need to strip the extra prompt from gen output
                    original_new_batch = self._union_gen_batch_output(original_new_batch, gen_batch_output)
                    augmented_new_batch = augmented_new_batch.union(gen_batch_output)

                    # Split batches based on train_infer_mode
                    if train_mode == "augmented":
                        new_train_batch = augmented_new_batch
                    else:
                        new_train_batch = original_new_batch
                    
                    if infer_mode == "augmented":
                        new_infer_batch = augmented_new_batch
                    else:
                        new_infer_batch = original_new_batch

                    with _timer("reward", timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_train_batch)
                            new_train_batch = new_train_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if not self.use_rm_worker:
                            reward_result = self.reward_fn(new_train_batch, return_dict=True)
                            reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result["reward_extra_info"]
                        else:
                            new_train_batch_padded, pad_size = pad_dataproto_to_divisor(new_train_batch, self.rm_worker_wg.world_size)
                            self._report_timing_stat()
                            reward_proto = self.rm_worker_wg.compute_reward(new_train_batch_padded)
                            reward_proto = unpad_dataproto(reward_proto, pad_size)
                            reward_tensor = reward_proto.batch['reward_tensor']
                            reward_extra_infos_dict = reward_proto.non_tensor_batch
                            self._report_timing_stat()


                        new_train_batch.batch["token_level_scores"] = reward_tensor


                        is_finish_list = None
                        if "is_finish" in reward_extra_infos_dict:
                            is_finish_list = reward_extra_infos_dict.pop("is_finish")
                            
                        print(f"reward_extra_infos_dict_keys:{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            new_train_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_train_batch, kl_metrics = apply_kl_penalty(
                                new_train_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(
                                kl_metrics
                            )  # TODO: This will be cleared if we use multiple genenration batches
                        else:
                            new_train_batch.batch["token_level_rewards"] = new_train_batch.batch["token_level_scores"]

                    if self.use_few_shot and "rollout_group_mapping" in augmented_new_batch.non_tensor_batch:

                        trajectory_rewards = new_train_batch.batch["token_level_rewards"].sum(dim=-1) # B, G

                        fewshot_metrics = self.few_shot_manager.compute_example_statistics(
                            rollout_mapping=augmented_new_batch.non_tensor_batch["rollout_group_mapping"],
                            rewards=trajectory_rewards,  
                            step=self.global_steps
                        )
                        metrics.update(fewshot_metrics)
                        self.few_shot_manager.update(
                            rollout_mapping=augmented_new_batch.non_tensor_batch["rollout_group_mapping"],
                            rewards=trajectory_rewards,
                        )
                    
                    if is_finish_list is not None:
                        assert len(is_finish_list) == len(new_train_batch.non_tensor_batch["uid"])
                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_train_batch.non_tensor_batch["uid"]):
                            if is_finish_list[idx] == 0:
                                continue
                            else:
                                kept_traj_idxs.append(idx)
                        pre_len = len(new_train_batch)
                        new_train_batch = new_train_batch[kept_traj_idxs]
                        new_infer_batch = new_infer_batch[kept_traj_idxs]
                        aft_len = len(new_train_batch)
                        print(f"pre_len:{pre_len},aft_len:{aft_len}")

                    if not self.config.algorithm.filter_groups.enable:
                        train_batch = new_train_batch
                        infer_batch = new_infer_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_train_batch.non_tensor_batch["seq_final_reward"] = (
                                new_train_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                            )
                        elif metric_name == "seq_reward":
                            new_train_batch.non_tensor_batch["seq_reward"] = (
                                new_train_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                            )

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(
                            new_train_batch.non_tensor_batch["uid"], new_train_batch.non_tensor_batch[metric_name]
                        ):  # We only need to ensure prompts from the same source share the same uid
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        if self.config.algorithm.filter_groups.adv:
                            # Filtering is done by uid uniformly, so no need to worry about misalignment
                            kept_prompt_uids = [
                                uid
                                for uid, std in prompt_uid2metric_std.items()
                                if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                            ]
                        else:
                            kept_prompt_uids = [
                                uid
                                for uid, std in prompt_uid2metric_std.items()
                            ]


                        if self.config.algorithm.filter_groups.avg is not None and self.config.algorithm.filter_groups.avg > 0:
                            prompt_uid2metric_avg = {}
                            for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                                prompt_uid2metric_avg[prompt_uid] = np.mean(metric_vals)

                            kept_prompt_uids_avg = [
                                uid
                                for uid, avg in prompt_uid2metric_avg.items()
                                if avg <= self.config.algorithm.filter_groups.avg
                            ]
                            
                            kept_prompt_uids = list(set(kept_prompt_uids) & set(kept_prompt_uids_avg))
                        
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_prompt_uids_set = set(kept_prompt_uids)
                        # Filter train_batch
                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_train_batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in kept_prompt_uids_set:
                                kept_traj_idxs.append(idx)
                        new_train_batch = new_train_batch[kept_traj_idxs]
                        
                        # Filter infer_batch
                        kept_infer_idxs = []
                        for idx, uid in enumerate(new_infer_batch.non_tensor_batch["uid"]):
                            if uid in kept_prompt_uids_set:
                                kept_infer_idxs.append(idx)
                        new_infer_batch = new_infer_batch[kept_infer_idxs]
                        
                        train_batch = new_train_batch if train_batch is None else DataProto.concat([train_batch, new_train_batch])
                        infer_batch = new_infer_batch if infer_batch is None else DataProto.concat([infer_batch, new_infer_batch]) 

                        prompt_bsz = self.config.data.train_batch_size
                        target_size = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                        if len(train_batch) < target_size:
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                continue
                            else:
                                world_size = self.actor_rollout_wg.world_size
                                max_batch_size = (train_batch.batch['input_ids'].shape[0] // world_size) * world_size
                                train_batch = train_batch[:max_batch_size]
                                infer_batch = infer_batch[:max_batch_size]
                                print(f"Raw Batch Size: {self.config.data.train_batch_size}, Retain Size: {num_prompt_in_batch}")
                        else:
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            train_batch = train_batch[:traj_bsz]
                            infer_batch = infer_batch[:traj_bsz]
                    
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        # Note: balancing train_batch will shuffle uid ordering; infer_batch is adjusted accordingly
                        self._balance_batch(train_batch, infer_batch, metrics=metrics)

                    # compute global_valid tokens
                    train_batch.meta_info["global_token_num"] = torch.sum(train_batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(infer_batch)
                        train_batch = train_batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(train_batch)
                            train_batch = train_batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(train_batch)
                            train_batch = train_batch.union(values)

                    with _timer("adv", timing_raw):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        train_batch = compute_advantage(
                            train_batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(train_batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)
                    
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(train_batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with _timer("testing", timing_raw):
                            self._report_timing_stat()
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=train_batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=train_batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=train_batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                train_batch = None
                infer_batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)
                self._report_timing_stat()

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1

                run_path = self.config.trainer.default_local_dir + os.sep + f"run_{self.global_steps}.log"
                os.makedirs(self.config.trainer.default_local_dir, exist_ok=True)
                with open(run_path, "w") as f:
                    try:
                        subprocess.run(
                            [shutil.which("ray"), "job", "logs", RAY_JOB_SUBMISSION_ID],
                            stdout=f,
                            stderr=subprocess.STDOUT,
                            check=True,
                            timeout=60,
                        )
                    except:
                        pass
