"""
Few-Shot Manager for augmenting prompts with example conversations.
This module is designed to be framework-agnostic and easily extensible.
"""

import re
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from verl import DataProto
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

import os

from verl.utils.fewshots.strategies import (
    SelectionStrategy,
    RandomStrategy,
)

@dataclass
class FewShotConfig:
    """Configuration for few-shot learning enhancement.
    
    Parameters:
        General:
            example_file: Path to the jsonl file containing example conversations
            num_fewshots: Number of examples (number of fewshots)
            num_groups: Number of groups of fewshots 
            selection_strategy: 'random'
            enabled: Whether to enable few-shot augmentation
    """
    example_file: Optional[str] = None
    num_fewshots: int = 3
    num_groups: int = 15
    selection_strategy: str = 'random'
    template_format: str = "multi-round"
    enabled: bool = False


class FewShotManager:
    """
    Manager for few-shot learning enhancement in RL training.
    
    This class handles:
    1. Loading example conversations from a jsonl file
    2. Selecting appropriate examples using random strategy
    3. Augmenting prompts with selected examples
    4. Managing statistics for example performance tracking
    """
    
    def __init__(self, config: FewShotConfig):
        """
        Initialize the Few-Shot Manager.
        
        Args:
            config: Configuration object for few-shot learning
        """
        self.config = config
        self.examples = []
        self.strategy = None

        # Performance tracking for each example
        self.example_stats = {}  # {example_idx: {'rewards': []}}
        
        if config.enabled and config.example_file:
            self._load_examples(config.example_file)
            self._initialize_strategy()
            # Initialize stats
            for i in range(len(self.examples)):
                self.example_stats[i] = {'rewards': []}

        print(f"Few-Shot Manager Config: {config}")
        
            
    def _load_examples(self, filepath: str):
        """
        Load example conversations from a jsonl file.
        
        Args:
            filepath: Path to the jsonl file
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                example = json.loads(line.strip())
                # Validate example format
                if 'query' in example and 'answer' in example:
                    self.examples.append(example)
                else:
                    print(f"Warning: Skipping invalid example: {example}")
        print(f"Loaded {len(self.examples)} few-shot examples from {filepath}")

    def _initialize_strategy(self):
        """
        Initialize the selection strategy based on config.
        """
        strategy_name = self.config.selection_strategy.lower()
        self.strategy_name = strategy_name
        
        if strategy_name == 'random':
            self.strategy = RandomStrategy(self.examples)
        else:
            raise ValueError(
                f"Unknown selection strategy: '{strategy_name}'. "
                f"Available strategies: random"
            )
            
    
    def _extract_last_user_message(self, formatted_text: str) -> str:
        """
        Extract the last user message from a formatted conversation.
        
        Args:
            formatted_text: Formatted conversation text (may include chat template tokens)
            
        Returns:
            The last user message content
        """
        patterns = [
            r'<｜User｜>(.*?)<｜Assistant｜><think>' # distill-qwen2.5
        ] 

        for p in patterns:
            matches = re.findall(p, formatted_text, re.DOTALL)
            if matches:
                return matches[-1].strip()

        raise ValueError("No patterns matched!")


    def format_conversation(self, examples: List[Dict], original_text: str, tokenizer, template_format: str="multi-round") -> str:
        """
        Format a multi-turn conversation with few-shot examples followed by the query.
        
        This method follows verl's standard conversation formatting approach:
        - Extracts the query from original_text
        - Builds a messages list with few-shot examples + query
        - Uses tokenizer's chat_template for proper formatting
        - Returns raw text (special tokens will be handled during tokenization)
        
        Args:
            examples: List of example conversations to include as few-shot demonstrations
                    Each example should have 'query' and 'answer' keys
            original_text: The original formatted conversation text (from decoded raw_prompt_ids)
            tokenizer: Tokenizer to use for formatting (must have apply_chat_template)
            template_format: Format to organize the examples. 
            
        Returns:
            Formatted conversation string ready for tokenization
        """
        if len(examples) == 0:
            return original_text
        
        # Step 1: Extract the actual query from original_text
        query = self._extract_last_user_message(original_text)
        
        # Step 2: Build messages list following format
        if template_format == "multi-round":
            messages = []
            for example in examples:
                messages.append({"role": "user", "content": example["query"]})
                messages.append({"role": "assistant", "content": example["answer"]})
            messages.append({"role": "user", "content": query})
            formatted = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            return formatted
        
        elif template_format == "in-context":
            demo_text = ""
            for i, example in enumerate(examples, 1):
                demo_text += f"Example {i}:\n"
                demo_text += f"Query: {example['query']}\n"
                demo_text += f"Answer: {example['answer']}\n\n"
            full_content = (
                f"You are an exceptional expert in mathematical reasoning and problem-solving. "
                f"Below are solutions from other solvers:\n\n"
                f"{demo_text}"
                f"You are more skilled than them. "
                f"Now demonstrate your superior ability by solving this problem even better:\n{query}"
            )
            messages = [{"role": "user", "content": full_content}]
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return formatted

        raise ValueError(f"Example format must be one of ['multi-round', 'in-context'], but get {template_format}!")


    def augment_batch_dict(
        self,
        batch_dict,
        num_rollouts_with_fewshots: int,
        num_rollouts_without_fewshots: int,
        max_prompt_length: int,
        truncation: str,
        tokenizer,
        template_format: str
    ):
        """
        Augment a batch_dict with few-shot examples for multiple rollouts.
        
        This creates num_rollouts (= num_rollouts_with_fewshots + num_rollouts_without_fewshots) 
        versions of each prompt:
        - Rollouts 0 to num_rollouts_with_fewshots-1: Include different few-shot examples
        - Rollouts num_rollouts_with_fewshots to num_rollouts-1: Keep original prompt (baseline)
        
        Args:
            batch_dict: Original batch dictionary with tensor and non-tensor data
            num_rollouts_with_fewshots: Number of rollouts per prompt that include few-shot examples
            num_rollouts_without_fewshots: Number of rollouts per prompt that keep the original prompt
            max_prompt_length: Maximum prompt length
            truncation: Truncation strategy ('left', 'right', or 'error')
            tokenizer: Tokenizer for formatting conversations
            
        Returns:
            Augmented batch_dict with repeated prompts (each with different few-shot examples)
        """
        total_rollouts = num_rollouts_with_fewshots + num_rollouts_without_fewshots

        # If no augmentation is requested, return original batch
        if not self.config.enabled or len(self.examples) == 0 or num_rollouts_with_fewshots == 0:
            return batch_dict

        # Get batch size
        batch_size = len(batch_dict['input_ids'])

        # Select example groups for this batch
        example_groups = self.strategy.select(
            num_groups=num_rollouts_with_fewshots,
            num_shots_per_group=self.config.num_fewshots
        )

        # Add empty groups for baseline rollouts
        example_groups += [[] for _ in range(num_rollouts_without_fewshots)]

        # Store original prompts for later recovery
        original_prompt_ids = batch_dict['raw_prompt_ids']  # array of list of int

        # Prepare lists for augmented batch
        all_augmented_input_ids = []
        all_augmented_attention_masks = []
        all_augmented_position_ids = []
        all_augmented_raw_prompt_ids = []
        rollout_group_mapping = []

        # Generate rollouts for each sample in the batch
        for sample_idx in range(batch_size):
            original_ids = original_prompt_ids[sample_idx]

            for rollout_idx, example_indices in enumerate(example_groups):

                if len(example_indices) > 0:
                    # Rollout with few-shot examples
                    selected_examples = [self.examples[idx] for idx in example_indices]

                    # Step 1: Decode original prompt to text
                    original_text = tokenizer.decode(original_ids, skip_special_tokens=False)

                    # Step 2: Insert few-shot examples
                    augmented_text = self.format_conversation(
                        selected_examples,
                        original_text,
                        tokenizer,
                        template_format
                    ) 
                    
                    # Step 3: Tokenize augmented prompt
                    augmented_model_inputs = tokenizer(
                        augmented_text,
                        return_tensors="pt",
                        add_special_tokens=False
                    )
                    augmented_input_ids = augmented_model_inputs.pop("input_ids")
                    augmented_attention_mask = augmented_model_inputs.pop("attention_mask")

                    # Step 4: Postprocess with verl_F (padding and truncation)
                    augmented_input_ids, augmented_attention_mask = verl_F.postprocess_data(
                        input_ids=augmented_input_ids,
                        attention_mask=augmented_attention_mask,
                        max_length=max_prompt_length,
                        pad_token_id=tokenizer.pad_token_id,
                        left_pad=True,
                        truncation=truncation,
                    )

                    # Step 5: Compute position_ids from attention_mask
                    augmented_position_ids = compute_position_id_with_mask(augmented_attention_mask)

                    # Step 6: Generate raw_prompt_ids separately
                    augmented_raw_prompt_ids = np.array(tokenizer.encode(augmented_text, add_special_tokens=False))
                    if len(augmented_raw_prompt_ids) > max_prompt_length:
                        if truncation == "left":
                            augmented_raw_prompt_ids = augmented_raw_prompt_ids[-max_prompt_length:]
                        elif truncation == "right":
                            augmented_raw_prompt_ids = augmented_raw_prompt_ids[:max_prompt_length]
                        elif truncation == "error":
                            raise RuntimeError(
                                f"Prompt length {len(augmented_raw_prompt_ids)} is longer than {max_prompt_length}."
                            )

                    augmented_input_ids = augmented_input_ids.squeeze(dim=0)
                    augmented_attention_mask = augmented_attention_mask.squeeze(dim=0)
                    augmented_position_ids = augmented_position_ids.squeeze(dim=0)

                    rollout_group_mapping.append((sample_idx, rollout_idx, example_indices))

                else:
                    # Baseline rollout: reuse original batch data
                    augmented_input_ids = batch_dict['input_ids'][sample_idx]
                    augmented_attention_mask = batch_dict['attention_mask'][sample_idx]
                    augmented_position_ids = batch_dict['position_ids'][sample_idx]
                    augmented_raw_prompt_ids = original_ids

                    rollout_group_mapping.append((sample_idx, rollout_idx, None))

                # Append to lists
                all_augmented_input_ids.append(augmented_input_ids)
                all_augmented_attention_masks.append(augmented_attention_mask)
                all_augmented_position_ids.append(augmented_position_ids)
                all_augmented_raw_prompt_ids.append(augmented_raw_prompt_ids)

        # Create augmented batch_dict (flat dictionary structure)
        augmented_batch_dict = {
            "input_ids": torch.stack(all_augmented_input_ids, dim=0),
            "attention_mask": torch.stack(all_augmented_attention_masks, dim=0),
            "position_ids": torch.stack(all_augmented_position_ids, dim=0),
            "raw_prompt_ids": np.array(all_augmented_raw_prompt_ids, dtype=object),
            "rollout_group_mapping": np.array(rollout_group_mapping, dtype=object),
        }

        # Repeat non-tensor metadata for each rollout
        for key in batch_dict.keys():
            if key not in augmented_batch_dict:  # Skip already handled keys
                if isinstance(batch_dict[key], torch.Tensor):
                    # Skip tensor keys that should not be repeated
                    continue
                else:
                    # Repeat non-tensor data for each rollout
                    augmented_batch_dict[key] = np.repeat(batch_dict[key], total_rollouts, axis=0)

        return augmented_batch_dict

    def compute_example_statistics(self, rollout_mapping, rewards, step):
        """Compute statistics for each example based on observed rewards.
        
        Args:
            rollout_mapping: The rollout_group_mapping from augmented batch.
                e.g.: [(prompt_id, rollout_id, example_indices)]
            rewards: Tensor or array of rewards for each trajectory
            step: Current training step
        
        Returns:
            Dictionary containing statistics for metrics logging
        """
        if not self.config.enabled or rollout_mapping is None:
            return {}
        
        # Convert rewards to numpy
        if torch.is_tensor(rewards):
            rewards = rewards.cpu().numpy()
        
        # Group rewards by prompt
        prompt_rewards = {}  # {prompt_idx: {'fewshot': [(reward, example_indices)], 'baseline': [rewards]}}
        
        for mapping_item, reward in zip(rollout_mapping, rewards):
            prompt_idx, rollout_idx, example_indices = mapping_item
            
            if prompt_idx not in prompt_rewards:
                prompt_rewards[prompt_idx] = {'fewshot': [], 'baseline': []}
            
            if example_indices is not None:  # few-shot
                prompt_rewards[prompt_idx]['fewshot'].append((reward, example_indices))
            else:  # baseline
                prompt_rewards[prompt_idx]['baseline'].append(reward)
        
        # Current step statistics
        current_step_stats = {}
        
        # Calculate reward statistics for each example
        for prompt_idx, data in prompt_rewards.items():
            for reward, example_indices in data['fewshot']:
                # Update global stats
                for idx in example_indices:
                    self.example_stats[idx]['rewards'].append(reward)
                    
                    # Update current step stats
                    if idx not in current_step_stats:
                        current_step_stats[idx] = {'rewards': []}
                    current_step_stats[idx]['rewards'].append(reward)
        
        # Prepare metrics using current step data only
        return self._prepare_metrics(step, current_step_stats)

    def _prepare_metrics(self, step, current_step_stats):
        """Prepare metrics for logging using current step data.
        
        Args:
            step: Current training step
            current_step_stats: Statistics from current step only
        """
        metrics = {}
        
        # Calculate mean reward for each example in current step
        mean_rewards = []
        
        for idx, stats in current_step_stats.items():
            if stats['rewards']:
                mean_rewards.append(np.mean(stats['rewards']))
        
        if not mean_rewards:
            return metrics
        
        # Distribution statistics for mean(r_i)
        metrics['train/example_reward_mean'] = np.mean(mean_rewards)
        metrics['train/example_reward_std'] = np.std(mean_rewards)
        metrics['train/example_reward_median'] = np.median(mean_rewards)
        metrics['train/example_reward_max'] = np.max(mean_rewards)
        metrics['train/example_reward_min'] = np.min(mean_rewards)
        
        return metrics


    def update(self, rollout_mapping, rewards):
        # No-op for random strategy
        pass
