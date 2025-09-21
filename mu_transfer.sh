#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=mu_transfer.out

eval "$(conda shell.bash hook)"
conda activate nanogpt

# show rich per-rank traces
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL   # or INFO

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
            torchrun --nproc_per_node=2 mu_transfer.py \
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

# now=$(date +%Y-%m-%d_%H-%M-%S)
# out_dir=mu_transfer_results/${now}
# mkdir -p ${out_dir}

# # Sweep settings from the table (fixed head_size=128, n_layers=3, n_kv_head=4)
# widths=(512 768 1024 2048)
# n_layers=3
# n_kv_head=4
# n_heads_list=(4 6 8 16)
# batch_list=(12 17 22 42)
# steps_list=(1160 1715 2356 4754)

# # learning rates: 2^{-4} to 2^{-8}
# lrs=(0.06250 0.03125 0.01563 0.00781 0.00391)
# weight_decays=(0.12062 0.24125 0.48250 0.96500 1.92999)

# param_types=(mup)

# for seed in {0..3}; do
#   for idx in ${!widths[@]}; do
#     width=${widths[$idx]}
#     n_heads=${n_heads_list[$idx]}
#     batch_size=${batch_list[$idx]}
#     steps=${steps_list[$idx]}
#     for lr in ${lrs[@]}; do
#         for wd in ${weight_decays[@]}; do
#             echo "width: ${width}, n_heads: ${n_heads}, n_kv_head: ${n_kv_head}, lr: ${lr}, wd: ${wd}, seed: ${seed}"
#             torchrun --nproc_per_node=2 mu_transfer.py \
#             --out_dir=${out_dir} \
#             --n_embd=${width} \
#             --n_layer=${n_layers} \
#             --n_head=${n_heads} \
#             --n_kv_head=${n_kv_head} \
#             --batch_size=${batch_size} \
#             --max_iters=${steps} \
#             --learning_rate=${lr} \
#             --weight_decay=${wd} \
#             --seed=${seed} \
#             --block_size=2048 \
#             --dropout=0.0 \
#             --bias=False \
#             --init_std=0.02 \
#             --beta1=0.9 \
#             --beta2=0.95 \
#             --grad_clip=1.0 \
#             --decay_lr=False \
#             --device='cuda' \
#             --dtype='bfloat16' \
#             --compile=False
#         done
#     done
#   done
# done