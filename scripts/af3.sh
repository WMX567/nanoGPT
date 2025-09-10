#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=32GB
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=af3ver9.out




eval "$(conda shell.bash hook)"
conda activate pyCLGL




python /scratch1/mengxiwu/3DPointDA/trainer_source_seg.py --dataroot /scratch1/mengxiwu/3DPointDA/PointSegDA/data/PointSegDAdataset --src_dataset adobe --trgt_dataset faust --seed 3 --model_type ver9
