#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=mu_transfer.out

eval "$(conda shell.bash hook)"
conda activate nanogpt

now=$(date +%Y-%m-%d_%H-%M-%S)
out_dir=mu_transfer_results/${now}
mkdir -p ${out_dir}

width=512
n_layers=3
n_kv_head=4
n_heads=4
batch_size=6
steps=1160
lr=0.06250
wd=0.12062
seed=0

echo "width: ${width}, n_heads: ${n_heads}, n_kv_head: ${n_kv_head}, lr: ${lr}, wd: ${wd}, seed: ${seed}"
            torchrun --standalone --nproc_per_node=2 mu_transfer.py \
            --out_dir=${out_dir} \
            --n_embd=${width} \
            --n_layer=${n_layers} \
            --n_head=${n_heads} \
            --n_kv_head=${n_kv_head} \
            --batch_size=${batch_size} \
            --max_iters=${steps} \
            --learning_rate=${lr} \
            --weight_decay=${wd} \
            --seed=${seed} \
            --block_size=1024 \
            --dropout=0.0 \
            --bias=False \
            --init_std=0.02 \
            --beta1=0.9 \
            --beta2=0.95 \
            --grad_clip=1.0 \
            --decay_lr=False \
            --device='cuda' \
            --dtype='bfloat16' \
            --compile=False
