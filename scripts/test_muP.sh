#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=32GB
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=af1ver9.out

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate nanogpt

export WANDB_API_KEY="bc5f2aabe268b0860305212d6f5e59a5ef296b4f"
wandb login --relogin "$WANDB_API_KEY"

export HF_DATASETS_TRUST_REMOTE_CODE=1

python /scratch1/mengxiwu/nanoGPT/data/openwebtext/prepare.py

python /scratch1/mengxiwu/nanoGPT/mup_paper_experiments/build_orchastrator.py \
  --config_generator_file mup_paper_experiments/configs/width_only.py \
  --max_concurrent 30 \
  --dry-run