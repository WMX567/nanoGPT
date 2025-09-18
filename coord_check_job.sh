#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=coord.out

eval "$(conda shell.bash hook)"
conda activate nanogpt

now=$(date +%Y-%m-%d_%H-%M-%S)
out_dir=coord-check-impl/${now}
mkdir -p ${out_dir}

head_size=32

embeds=(192 384 768 1536)
head_sizes=(32 64 128 256)
for seed in {0..3}; do
    for head_size in "${head_sizes[@]}"; do
        for emb in "${embeds[@]}"; do
            mup_multiplier=$((emb / 192))
            n_heads=$((emb / head_size ))
            n_kv_heads=6
            # n_kv_heads is random number between 2 and n_heads
            # n_kv_heads=$(( RANDOM % (n_heads - 1) + 2 ))
            echo "mup_muliplier: ${mup_multiplier}, n_heads: ${n_heads}, emb: ${emb}, seed: ${seed}"
            srun python coord_check.py \
                --out_dir=${out_dir} \
                --eval_interval=10000000 \
                --log_interval=10000000 \
                --eval_iters=1 \
                --eval_only=False \
                --init_from='scratch' \
                --wandb_log=False \
                --dataset='openwebtext' \
                --gradient_accumulation_steps=1 \
                --batch_size=1 \
                --block_size=1024 \
                --n_layer=3 \
                --n_head=${n_heads} \
                --n_kv_head=${n_kv_heads} \
                --n_embd=${emb} \
                --dropout=0.0 \
                --bias=False \
                --init_std=0.02 \
                --learning_rate=4e-5 \
                --max_iters=4 \
                --weight_decay=0.0 \
                --beta1=0.9 \
                --beta2=0.95 \
                --grad_clip=1.0 \
                --decay_lr=False \
                --mup=True \
                --mup_multiplier=${mup_multiplier} \
                --seed=${seed} \
                --backend='nccl' \
                --device='cuda' \
                --dtype='float32' \
                --compile=False \
                --impl='mengxi_impl'
        done
    done
done