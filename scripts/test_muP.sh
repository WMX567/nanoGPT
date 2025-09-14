#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --output=test.out

eval "$(conda shell.bash hook)"
conda activate nanogpt

python /scratch1/mengxiwu/nanoGPT/mup_paper_experiments/build_orchastrator.py \
  --config_generator_file /scratch1/mengxiwu/nanoGPT/mup_paper_experiments/configs/width_only.py \
  --max_concurrent 1 \
  --dry_run