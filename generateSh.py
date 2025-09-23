import os
import itertools
from datetime import datetime

def generate_sh_scripts():

    widths = [512, 768, 1024, 2048]
    n_layers = 3
    n_kv_head = 2
    n_heads_list = [4, 6, 8, 16]
    batch_list = [12, 17, 22, 22]
    steps_list = [1160, 1715, 2356, 4754]
    
    # learning rates: 2^{-6} to 2^{-12}
    lrs = [0.01563, 0.00781, 0.00391, 0.00195, 0.00098, 0.00049, 0.00024]
    seeds = [0, 1, 2]
    
    output_dir = "generated_scripts"
    os.makedirs(output_dir, exist_ok=True)
    
    script_count = 0
    
    for seed in seeds:
        for idx, width in enumerate(widths):
            n_heads = n_heads_list[idx]
            batch_size = batch_list[idx]
            steps = steps_list[idx]
            
            for lr in lrs:

                wd = 1/steps/lr/0.035
                script_count += 1
                script_name = f"mu_transfer_w{width}_h{n_heads}_lr{lr:.5f}_wd{wd:.5f}_s{seed}.sh"
                script_path = os.path.join(output_dir, script_name)

                with open(script_path, 'w') as f:
                    f.write('#!/bin/bash\n')
                    f.write('#SBATCH --partition=gpu\n')
                    f.write('#SBATCH --time=12:00:00\n')
                    f.write('#SBATCH --gres=gpu:1\n')
                    f.write('#SBATCH --mem=64G\n')
                    f.write('#SBATCH --ntasks-per-node=1\n')
                    f.write('#SBATCH --cpus-per-task=4\n')
                    f.write(f'#SBATCH --output=mu_transfer_w{width}_h{n_heads}_lr{lr:.5f}_wd{wd:.5f}_s{seed}.out\n')
                    f.write('\n')
                    f.write('eval "$(conda shell.bash hook)"\n')
                    f.write('conda activate nanogpt\n')
                    f.write('\n')
                    f.write(f'width={width}\n')
                    f.write(f'n_layers={n_layers}\n')
                    f.write(f'n_kv_head={n_kv_head}\n')
                    f.write(f'n_heads={n_heads}\n')
                    f.write(f'batch_size={batch_size}\n')
                    f.write(f'steps={steps}\n')
                    f.write(f'lr={lr:.5f}\n')
                    f.write(f'wd={wd:.5f}\n')
                    f.write(f'seed={seed}\n')
                    f.write('\n')
                    f.write('out_dir=mu_transfer_results\n')
                    f.write('mkdir -p ${out_dir}\n')
                    f.write('\n')
                    f.write('echo "Starting training with parameters:"\n')
                    f.write('echo "width: ${width}, n_heads: ${n_heads}, n_kv_head: ${n_kv_head}"\n')
                    f.write('echo "lr: ${lr}, wd: ${wd}, seed: ${seed}"\n')
                    f.write('echo "batch_size: ${batch_size}, steps: ${steps}"\n')
                    f.write('echo "output_dir: ${out_dir}"\n')
                    f.write('\n')
                    f.write('python /scratch1/mengxiwu/nanoGPT/mu_transfer.py \\\n')
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
                    f.write('    --init_std=0.02 \\\n')
                    f.write('    --beta1=0.9 \\\n')
                    f.write('    --beta2=0.95 \\\n')
                    f.write('    --grad_clip=1.0 \\\n')
                    f.write('    --decay_lr=False \\\n')
                    f.write('    --device="cuda:0" \\\n')
                    f.write('    --dtype="bfloat16" \\\n')
                    f.write('    --compile=False\n')
                    f.write('\n')
                    f.write('echo "Training completed for w${width}_h${n_heads}_lr${lr}_wd${wd}_s${seed}"\n')
                
    submit_all_script = os.path.join(output_dir, "submit_all.sh")
    with open(submit_all_script, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('# This script submits all generated jobs to SLURM\n')
        f.write('\n')
        for seed in seeds:
            for idx, width in enumerate(widths):
                n_heads = n_heads_list[idx]
                for lr in lrs:
                    wd = 1/steps_list[idx]/lr/0.035
                    script_name = f"mu_transfer_w{width}_h{n_heads}_lr{lr:.5f}_wd{wd:.5f}_s{seed}.sh"
                    f.write(f'sbatch {script_name}\n')
        f.write('\n')
        f.write(f'echo "Submitted {script_count} jobs to SLURM"\n')
    
    print(f"Generated {script_count} scripts and a submit_all.sh script in {output_dir}")

if __name__ == "__main__":
    generate_sh_scripts()