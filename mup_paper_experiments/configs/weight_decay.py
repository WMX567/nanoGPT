# Weight decay ablation experiment
# python mup_paper_experiments/build_orchastrator.py --config_generator_file mup_paper_experiments/configs/weight_decay.py --max_concurrent 30
# python crawl_wandb.py --entity krchickering-uc-davis --project wd-sweep-new --output-dir ./mup_paper_experiments/results/ablations/wd-sweep-new
# python crawl_wandb.py --entity krchickering-uc-davis --project wd-sweep-3 --output-dir ./mup_paper_experiments/results/ablations/wd-sweep-
# python crawl_wandb.py --entity krchickering-uc-davis --project wd-sweep-4 --output-dir ./mup_paper_experiments/results/ablations/wd-sweep-4



import numpy as np
from copy import deepcopy

PROD = True

MODEL_DEPTH = 8

LEARNING_RATE_SAMPLES = 5 if PROD else 1
learning_rates = [
    10**p for p in np.linspace(-3.5, -2, LEARNING_RATE_SAMPLES)
]
WEIGHT_DECAY_SAMPLES = 11 if PROD else 1
weight_decays = [
    10**p for p in np.linspace(-2, .5, WEIGHT_DECAY_SAMPLES)
]

seeds = [42, 43, 44] if PROD else [42]

WANDB_PROJECT = 'wd-sweep-4'

shared_params = {
    'log_wandb': 'true',
    'wandb_project': WANDB_PROJECT,
    'eval_interval': 50,
    'eval_iters': 100,
    'decay_profile': 'cosine',
    'decay_lr': 'true',
    'n_layer': MODEL_DEPTH,
    'avg_interval': 120,
}

# model_configs = [
#     {'n_embd': 384,  'n_head': 6,   'n_kv_head': 6,  'n_gpus': 2, 'gradient_accumulation_steps': 2,  'batch_size': 61, 'max_iters': 709 },
#     {'n_embd': 512,  'n_head': 8,   'n_kv_head': 8,  'n_gpus': 2, 'gradient_accumulation_steps': 2,  'batch_size': 63, 'max_iters': 874 },
#     {'n_embd': 2048, 'n_head': 32,  'n_kv_head': 32, 'n_gpus': 6, 'gradient_accumulation_steps': 6,  'batch_size': 66, 'max_iters': 2756, 'sbatch_mem': 256},
# ]

# Constant data. Keeps training curves "stacked"
model_configs = [
    {'n_embd': 384,  'n_head': 6,   'n_kv_head': 6,  'n_gpus': 2, 'gradient_accumulation_steps': 2,  'batch_size': 104, 'max_iters': 1250 },
    {'n_embd': 512,  'n_head': 8,   'n_kv_head': 8,  'n_gpus': 2, 'gradient_accumulation_steps': 2,  'batch_size': 104, 'max_iters': 1250 },
    # {'n_embd': 2048, 'n_head': 32,  'n_kv_head': 32, 'n_gpus': 2, 'gradient_accumulation_steps': 2,  'batch_size': 104, 'max_iters': 1250, 'sbatch_mem': 256},
]

configs = []
for mup in [True, False]:
    for lr in learning_rates:
        for wd in weight_decays:
            for seed in seeds:
                for conf in model_configs:
                    conf = deepcopy(conf)
                    conf['learning_rate'] = lr
                    conf['weight_decay'] = wd
                    conf['seed'] = seed

                    conf['min_lr'] = lr / 10

                    if mup:
                        conf['mup'] = 'true'
                        conf['mup_multiplier'] = float(conf['n_embd'] / 384)
                        conf['impl'] = 'tpv_left_impl'
                    else:
                        conf['mup'] = 'false'
                        conf['mup_multiplier'] = 1.0
                        conf['impl'] = 'standard_param_impl'

                    conf.update(shared_params)
                    configs.append(conf)

if __name__ == "__main__":
    import json
    import random

    # random.shuffle(configs)

    config_strings = []
    for config in configs:
        config_strings.append(json.dumps(config))

    print('\n'.join(config_strings))
