## Running Instructions

### Environment Setup

Install the required Python dependencies by executing the following command:

```
pip install -r requirements.txt
```

### Data Preparation

To prepare the real-world datasets, please download the **open-source** [LaDe dataset](https://huggingface.co/datasets/Cainiao-AI/LaDe/tree/main). 

1. Download both the `pickup` and `delivery` folders.
2. Place them into the `data/LADE/` directory of this project.
3. Run the following command to obtain the processed data:

```
python ./data/LADE/lade_process.py --city hz [--draw]
```


**Arguments:**
- `--city`: Specifies the target city dataset to be processed.
- `--draw`: (Optional) Generates a 2D distribution plot of the nodes and takeoff/landing stations.

### Training

For the synthetic dataset, use the following commands:

```
python train_dmpdp.py --env_config_path envs/config/synthetic_small.yaml --num_episodes 8192 --max_grad_norm 5 --n_rollout_threads 64 --mini_batch_size 256 --lr 3e-5 --eval_episodes 64
```

```
python train_dmpdp.py --env_config_path envs/config/synthetic_large.yaml --num_episodes 8192 --max_grad_norm 2 --n_rollout_threads 64 --mini_batch_size 64 --lr 3e-5 --eval_episodes 64
```

For the real-world dataset, use the following command:

```
python train_dmpdp.py --env_config_path envs/config/lade.yaml --num_episodes 8192 --max_grad_norm 1 --n_rollout_threads 64 --mini_batch_size 64 --lr 2e-5 --eval_episodes 64 --dataset hz
```

### Testing

For testing on the synthetic dataset, use the following commands (replace `...` with your actual model path):

```
python train_dmpdp.py --eval_episodes 64 --env_config_path './envs/config/synthetic_small.yaml' --model_path "..." --only_eval
```

```
python train_dmpdp.py --eval_episodes 64 --env_config_path './envs/config/synthetic_large.yaml' --model_path "..." --only_eval
```

For testing on the real-world dataset, use the following command (replace `...` with your actual model path):

```
python train_dmpdp.py --eval_episodes 64 --env_config_path './envs/config/lade.yaml' --model_path "..." --only_eval --dataset hz
```

## Structure

```
MoE-Pointer/
├── algorithms/        # Algorithm implementations
├── config.py          # Global configuration parameters
├── data/              # Data loading and processing utilities
├── envs/              # RL environment definitions and settings
├── runner/            # RL training runners
├── utils/             # Helper functions and RL utilities
└── train_dmpdp.py     # Main entry point for training
```

