#!/bin/bash
#SBATCH --time=14:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

srun python run_lambda_sweep.py --seed 5

# now=$(date +%Y-%m-%d_%H-%M-%S)
# out_dir=mu-transfer-char/${now}
# mkdir -p $out_dir

# head_size=64

# embeds=(256 512 1024 2048 4096)
# # [10**p for p in np.linspace(-5, 1, 15)]
# etas=(0.000051794746792312125 0.0002682695795279727 0.0013894954943731374 0.0071968567300115215 0.037275937203149416)
# # such that \eta \lambda = lambda_mult
# lambda_mults=(0.000005179474679231212 0.000026826957952797274 0.00013894954943731373)
    
# for seed in {0..5}; do
#     for emb in "${embeds[@]}"; do
#         for eta in "${etas[@]}"; do
#             for lambda_m in "${lambda_mults[@]}"; do
#                 mup_multiplier=$(( emb / 256 ))
#                 n_heads=$(( emb / head_size ))
                
#                 echo "eta: ${eta}, lambda_m: ${lambda_m}" # Debugging line
#                 lambda=$(echo "scale=16; $lambda_m / $eta" | bc)

#                 echo "mup_multiplier: ${mup_multiplier}, n_heads: ${n_heads}, emb: ${emb}, seed: ${seed}, eta: ${eta}, lambda: ${lambda}"
#                 srun python hp_train.py \
#                     --out_dir=${out_dir} \
#                     --wandb_log=True \
#                     --wandb_project='mu-transfer-char' \
#                     --eval_interval=500 \
#                     --log_interval=1 \
#                     --eval_iters=50 \
#                     --eval_only=False \
#                     --init_from='scratch' \
#                     --dataset='shakespeare_char' \
#                     --gradient_accumulation_steps=1 \
#                     --batch_size=256 \
#                     --block_size=1024 \
#                     --n_layer=3 \
#                     --n_head=${n_heads} \
#                     --n_embd=${emb} \
#                     --dropout=0.0 \
#                     --bias=False \
#                     --init_std=0.02 \
#                     --learning_rate=${eta} \
#                     --max_iters=500 \
#                     --weight_decay=${lambda} \
#                     --beta1=0.9 \
#                     --beta2=0.95 \
#                     --grad_clip=1.0 \
#                     --decay_lr=False \
#                     --mup=True \
#                     --mup_multiplier=${mup_multiplier} \
#                     --seed=${seed} \
#                     --backend='nccl' \
#                     --device='cuda' \
#                     --dtype='float32' \
#                     --compile=False \
#                     --coord_check=False
#             done
#         done
#     done
# done

