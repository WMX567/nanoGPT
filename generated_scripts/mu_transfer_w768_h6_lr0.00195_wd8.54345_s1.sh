#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=mu_transfer_w768_h6_lr0.00195_wd8.54345_s1.out

eval "$(conda shell.bash hook)"
conda activate nanogpt

width=768
n_layers=3
n_kv_head=2
n_heads=6
batch_size=17
steps=1715
lr=0.00195
wd=8.54345
seed=1

out_dir=mu_transfer_results
mkdir -p ${out_dir}

echo "Starting training with parameters:"
echo "width: ${width}, n_heads: ${n_heads}, n_kv_head: ${n_kv_head}"
echo "lr: ${lr}, wd: ${wd}, seed: ${seed}"
echo "batch_size: ${batch_size}, steps: ${steps}"
echo "output_dir: ${out_dir}"

python /scratch1/mengxiwu/nanoGPT/mu_transfer.py \
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
    --init_std=0.02 \
    --beta1=0.9 \
    --beta2=0.95 \
    --grad_clip=1.0 \
    --decay_lr=False \
    --device="cuda:0" \
    --dtype="bfloat16" \
    --compile=False

echo "Training completed for w${width}_h${n_heads}_lr${lr}_wd${wd}_s${seed}"
