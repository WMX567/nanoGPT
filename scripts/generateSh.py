#!/usr/bin/env python3
"""
生成多个独立的 .sh 脚本文件，每个脚本运行一组特定的参数
避免在单个脚本中循环运行多组参数
"""

import os
import itertools
from datetime import datetime

def generate_sh_scripts():

    widths = [512, 1024, 2048]
    n_layers = 3
    n_kv_head = 4
    n_heads_list = [4, 8, 16]
    batch_list = [12, 22, 42]
    steps_list = [1160, 2356, 4754]
    
    # learning rates: 2^{-4} to 2^{-8}
    lrs = [0.06250, 0.03125, 0.01563, 0.00781, 0.00391]
    weight_decays = [0.12062, 0.24125, 0.48250, 0.96500, 1.92999]
    
    seeds = [0, 1, 2]
    
    output_dir = "../generated_scripts"
    os.makedirs(output_dir, exist_ok=True)
    
    script_count = 0
    
    for seed in seeds:
        for idx, width in enumerate(widths):
            n_heads = n_heads_list[idx]
            batch_size = batch_list[idx]
            steps = steps_list[idx]
            
            for lr, wd in zip(lrs, weight_decays):
                script_count += 1
                script_name = f"mu_transfer_w{width}_h{n_heads}_lr{lr:.5f}_wd{wd:.5f}_s{seed}.sh"
                script_path = os.path.join(output_dir, script_name)

                # 计算 gradient accumulation steps
                grad_accum_steps = 5 * 8 // (batch_size // 12)

                script_content = f"""#!/bin/bash
                #SBATCH --partition=gpu
                #SBATCH --time=08:00:00
                #SBATCH --gres=gpu:1
                #SBATCH --ntasks-per-node=1
                #SBATCH --cpus-per-task=4
                #SBATCH --output=mu_transfer_w{width}_h{n_heads}_lr{lr:.5f}_wd{wd:.5f}_s{seed}.out
                #SBATCH --job-name=mu_w{width}_s{seed}

                eval "$(conda shell.bash hook)"
                conda activate nanogpt

                # 参数设置
                width={width}
                n_layers={n_layers}
                n_kv_head={n_kv_head}
                n_heads={n_heads}
                batch_size={batch_size}
                steps={steps}
                lr={lr:.5f}
                wd={wd:.5f}
                seed={seed}
                grad_accum_steps={grad_accum_steps}

                out_dir=mu_transfer_results/w${{width}}_h${{n_heads}}_lr${{lr}}_wd${{wd}}_s${{seed}}
                mkdir -p ${{out_dir}}

                echo "Starting training with parameters:"
                echo "width: ${{width}}, n_heads: ${{n_heads}}, n_kv_head: ${{n_kv_head}}"
                echo "lr: ${{lr}}, wd: ${{wd}}, seed: ${{seed}}"
                echo "batch_size: ${{batch_size}}, steps: ${{steps}}"
                echo "grad_accum_steps: ${{grad_accum_steps}}"
                echo "output_dir: ${{out_dir}}"

                python /scratch1/mengxiwu/nanoGPT/mu_transfer.py \\
                    --out_dir=${{out_dir}} \\
                    --n_embd=${{width}} \\
                    --n_layer=${{n_layers}} \\
                    --n_head=${{n_heads}} \\
                    --gradient_accumulation_steps=${{grad_accum_steps}} \\
                    --n_kv_head=${{n_kv_head}} \\
                    --batch_size=${{batch_size}} \\
                    --max_iters=${{steps}} \\
                    --learning_rate=${{lr}} \\
                    --weight_decay=${{wd}} \\
                    --seed=${{seed}} \\
                    --block_size=1024 \\
                    --dropout=0.0 \\
                    --bias=False \\
                    --init_std=0.02 \\
                    --beta1=0.9 \\
                    --beta2=0.95 \\
                    --grad_clip=1.0 \\
                    --decay_lr=False \\
                    --device='cuda:0' \\
                    --dtype='bfloat16' \\
                    --compile=False

                echo "Training completed for w${{width}}_h${{n_heads}}_lr${{lr}}_wd${{wd}}_s${{seed}}"
                """
                
                # 写入脚本文件
                with open(script_path, 'w') as f:
                    f.write(script_content)
                
                print(f"Generated: {script_name}")

    submit_all_script = os.path.join(output_dir, "submit_all.sh")
    with open(submit_all_script, 'w') as f:
        f.write("#!/bin/bash\n")
        
        for seed in seeds:
            for idx, width in enumerate(widths):
                n_heads = n_heads_list[idx]
                for lr, wd in zip(lrs, weight_decays):
                    script_name = f"mu_transfer_w{width}_h{n_heads}_lr{lr:.5f}_wd{wd:.5f}_s{seed}.sh"
                    f.write(f"sbatch {script_name}\n")

        f.write(f"\necho 'Submitted {script_count} jobs to SLURM'\n")
    
    # os.chmod(submit_all_script, 0o755)
    
    # stats_script = os.path.join(output_dir, "job_stats.sh")
    # with open(stats_script, 'w') as f:
    #     f.write("""#!/bin/bash
    #     echo "Job Status Summary:"
    #     squeue -u $USER | grep mu_ | awk '{print $5}' | sort | uniq -c
    #     echo ""
    #     echo "Total jobs in queue:"
    #     squeue -u $USER | grep mu_ | wc -l
    #     echo ""
    #     echo "Running jobs:"
    #     squeue -u $USER | grep mu_ | grep " R " | wc -l
    #     echo ""
    #     echo "Pending jobs:"
    #     squeue -u $USER | grep mu_ | grep " PD " | wc -l
    #     """)
    
    # os.chmod(stats_script, 0o755)
    
    # print(f"\n总共生成了 {script_count} 个脚本文件")
    # print(f"脚本保存在: {output_dir}")
    # print(f"使用 'bash {submit_all_script}' 批量提交所有作业")
    # print(f"使用 'bash {stats_script}' 查看作业状态")

if __name__ == "__main__":
    generate_sh_scripts()
