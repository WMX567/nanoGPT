import os

# 创建输出目录
output_dir = 'mu_transfer_scripts'
os.makedirs(output_dir, exist_ok=True)

# Sweep settings from the mu_transfer.sh (fixed head_size=128, n_layers=3, n_kv_head=4)
widths = [512, 768, 1024, 2048]
n_layers = 3
n_kv_head = 4
n_heads_list = [4, 6, 8, 16]
batch_list = [12, 17, 22, 42]
steps_list = [1160, 1715, 2356, 4754]

# learning rates: 2^{-4} to 2^{-8}
lrs = [0.06250, 0.03125, 0.01563, 0.00781, 0.00391]
weight_decays = [0.12062, 0.24125, 0.48250, 0.96500, 1.92999]

seeds = [0, 1, 2, 3]

script_counter = 0
script_list = []

for seed in seeds:
    for idx in range(len(widths)):
        width = widths[idx]
        n_heads = n_heads_list[idx]
        batch_size = batch_list[idx]
        steps = steps_list[idx]
        for lr in lrs:
            for wd in weight_decays:
                script_counter += 1
                script_name = f'mu_transfer_w{width}_lr{lr:.5f}_wd{wd:.5f}_s{seed}.sh'
                script_path = os.path.join(output_dir, script_name)
                script_list.append(script_name)
                
                with open(script_path, 'w') as f:
                    f.write('#!/bin/bash\n')
                    f.write('#SBATCH --partition=gpu\n')
                    f.write('#SBATCH --time=20:00:00\n')
                    f.write('#SBATCH --gres=gpu:1\n')
                    f.write('#SBATCH --ntasks-per-node=1\n')
                    f.write('#SBATCH --cpus-per-task=4\n')
                    f.write(f'#SBATCH --output=mu_transfer_w{width}_lr{lr:.5f}_wd{wd:.5f}_s{seed}.out\n')
                    f.write('\n')
                    f.write('eval "$(conda shell.bash hook)"\n')
                    f.write('conda activate nanogpt\n')
                    f.write('\n')
                    f.write('now=$(date +%Y-%m-%d_%H-%M-%S)\n')
                    f.write('out_dir=mu_transfer_results/${now}\n')
                    f.write('mkdir -p ${out_dir}\n')
                    f.write('\n')
                    f.write(f'width={width}\n')
                    f.write(f'n_layers={n_layers}\n')
                    f.write(f'n_kv_head={n_kv_head}\n')
                    f.write(f'n_heads={n_heads}\n')
                    f.write(f'batch_size={batch_size}\n')
                    f.write(f'steps={steps}\n')
                    f.write(f'lr={lr}\n')
                    f.write(f'wd={wd}\n')
                    f.write(f'seed={seed}\n')
                    f.write('\n')
                    f.write('echo "width: ${width}, n_heads: ${n_heads}, n_kv_head: ${n_kv_head}, lr: ${lr}, wd: ${wd}, seed: ${seed}"\n')
                    f.write('python mu_transfer.py \\\n')
                    f.write('    --out_dir=${out_dir} \\\n')
                    f.write('    --n_embd=${width} \\\n')
                    f.write('    --n_layer=${n_layers} \\\n')
                    f.write('    --n_head=${n_heads} \\\n')
                    f.write('    --n_kv_head=${n_kv_head} \\\n')
                    f.write('    --batch_size=${batch_size} \\\n')
                    f.write('    --max_iters=${steps} \\\n')
                    f.write('    --learning_rate=${lr} \\\n')
                    f.write('    --weight_decay=${wd} \\\n')
                    f.write('    --seed=${seed} \\\n')
                    f.write('    --block_size=1024 \\\n')
                    f.write('    --dropout=0.0 \\\n')
                    f.write('    --bias=False \\\n')
                    f.write('    --init_std=0.02 \\\n')
                    f.write('    --beta1=0.9 \\\n')
                    f.write('    --beta2=0.95 \\\n')
                    f.write('    --grad_clip=1.0 \\\n')
                    f.write('    --decay_lr=False \\\n')
                    f.write('    --device="cuda:0" \\\n')
                    f.write('    --dtype="bfloat16" \\\n')
                    f.write('    --compile=False\n')
                
                # Make script executable
                os.chmod(script_path, 0o755)

print(f"Generated {script_counter} scripts in {output_dir}/ directory")

# 创建一个主提交脚本
with open(f'{output_dir}/submit_all.sh', 'w') as f:
    f.write('#!/bin/bash\n')
    f.write('# Submit all mu_transfer jobs\n')
    f.write('\n')
    for script_name in script_list:
        f.write(f'sbatch {script_name}\n')

os.chmod(f'{output_dir}/submit_all.sh', 0o755)

print(f"Created submit_all.sh to submit all {script_counter} jobs")
print(f"To run all jobs: cd {output_dir} && bash submit_all.sh")
