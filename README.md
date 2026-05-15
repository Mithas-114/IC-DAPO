# IC-DAPO: In-Context RLVR

Code for the paper *"Good Reasoning Makes Good Demonstrations: Implicit Reasoning Quality Supervision via In-Context Reinforcement Learning"*.

Based on the [verl](https://github.com/volcengine/verl) framework.

## Project Structure

```
IC-DAPO/
├── datasets/
│   ├── train/
│   │   ├── train_math_30K.json          # 30K training set (DAPO baseline)
│   │   ├── train_math_28K.jsonl         # 28K training set (IC-DAPO, after splitting E)
│   │   ├── demonstrations_r1.jsonl      # 1,082 demonstration examples (DeepSeek-R1)
│   │   └── demonstrations_v3.jsonl      # 1,082 demonstration examples (DeepSeek-V3.1)
│   └── eval/
├── verl/                                # Modified verl framework
│   ├── recipe/dapo/
│   │   ├── run/                         # Launch scripts
│   │   └── src/                         # IC-DAPO trainer & config
│   ├── multi_node_launcher.sh           # Multi-node Ray cluster launcher
│   ├── multi_node_stop.sh               # Multi-node Ray cluster stopper
│   └── verl/utils/fewshots/             # Demonstration manager (core IC-DAPO logic)
├── requirements1.txt                    # PyTorch & CUDA (install first)
└── requirements2.txt                    # All other dependencies
```

## Environment Setup

Summary:

```bash
conda create -n icdapo python=3.10 -y && conda activate icdapo

# Step 1: PyTorch
pip install -r requirements1.txt

# Step 2: Dependencies (after removing flash_attn and verl lines from requirements2.txt)
pip install -r requirements2.txt

# Step 3: flash_attn (must be after torch)
pip install flash_attn==2.7.4.post1 --no-build-isolation

# Step 4: verl from local source
cd verl && pip install -e .
```

## Important: All paths must be absolute

All shell scripts use a `PROJECT_ROOT` variable that auto-resolves to the project root. If auto-detection fails, set it explicitly before launching:

```bash
export PROJECT_ROOT=/absolute/path/to/IC-DAPO
```

Set `YOUR_MODEL_PATH` in each script to your local model path or HuggingFace model ID.

## Quick Start

### Step 1: Single-node debug (1 node, 8 GPUs)

Start a local Ray cluster and run the debug script:

```bash
ray start --head --dashboard-host=0.0.0.0 --dashboard-port=8265

export RAY_ADDRESS="http://localhost:8265"
export WORKING_DIR="${PROJECT_ROOT}/verl"

bash verl/recipe/dapo/run/run_icdapo_r1_1.5b_debug.sh
```

### Step 2: Multi-node training

Launch the Ray cluster across nodes:

```bash
bash verl/multi_node_launcher.sh MASTER_IP WORKER_IP1 WORKER_IP2 ...
```

Then submit the training job from any machine:

```bash
export RAY_ADDRESS="http://MASTER_IP:8265"
export WORKING_DIR="${PROJECT_ROOT}/verl"

# IC-DAPO with R1 demonstrations (1.5B, 8 nodes)
bash verl/recipe/dapo/run/run_icdapo_r1_1.5b.sh

# IC-DAPO with R1 demonstrations (7B, 8 nodes)
bash verl/recipe/dapo/run/run_icdapo_r1_7b.sh
```

To stop the cluster:

```bash
bash verl/multi_node_stop.sh MASTER_IP WORKER_IP1 WORKER_IP2 ...
```

## Available Scripts

| Script | Model | Method | Nodes |
|--------|-------|--------|-------|
| `run_dapo_1.5b.sh` | 1.5B | DAPO baseline | 4 |
| `run_dapo_7b.sh` | 7B | DAPO baseline | 16 |
| `run_icdapo_r1_1.5b.sh` | 1.5B | IC-DAPO (R1 demos) | 8 |
| `run_icdapo_r1_7b.sh` | 7B | IC-DAPO (R1 demos) | 8 |
| `run_icdapo_v3_1.5b.sh` | 1.5B | IC-DAPO (V3.1 demos) | 8 |
| `run_icdapo_v3_7b.sh` | 7B | IC-DAPO (V3.1 demos) | 8 |
| `run_icdapo_r1_1.5b_debug.sh` | 1.5B | IC-DAPO debug | 1 |

## Citation

```bibtex
@article{mei2025good,
  title={Good Reasoning Makes Good Demonstrations: Implicit Reasoning Quality Supervision via In-Context Reinforcement Learning},
  author={Mei, Tiehua and Lv, Minxuan and Pan, Leiyu and Su, Zhenpeng and Hou, Hongru and Chen, Hengrui and Xu, Ao and Yang, Deqing},
  year={2025}
}
```
