# Config for varying width only
import numpy as np
from copy import deepcopy

MODEL_DEPTH = 8

LEARNING_RATE_SAMPLES = 12
learing_rates = [
    10**p for p in np.linspace(-4.25, -2.5, LEARNING_RATE_SAMPLES)
]
WEIGHT_DECAY_SAMPLES = 12
weight_decays = [
    10**p for p in np.linspace(-12, 1, WEIGHT_DECAY_SAMPLES)
]

seeds = [42] #, 43, 44]

WANDB_PROJECT = 'wd-sweep'

model_configs = [
    # {'n_embd': 384,  'n_head': 6,   'n_kv_head': 6,  'n_layer': MODEL_DEPTH, 'log_wandb': 'true', 'wandb_project': WANDB_PROJECT, 'n_gpus': 3,                    'gradient_accumulation_steps': 3,  'batch_size': 54, 'max_iters': 635 },
    # {'n_embd': 512,  'n_head': 8,   'n_kv_head': 8,  'n_layer': MODEL_DEPTH, 'log_wandb': 'true', 'wandb_project': WANDB_PROJECT, 'n_gpus': 3,                    'gradient_accumulation_steps': 3,  'batch_size': 66, 'max_iters': 782 },
    {'n_embd': 2048, 'n_head': 32,  'n_kv_head': 32, 'n_layer': MODEL_DEPTH, 'log_wandb': 'true', 'wandb_project': WANDB_PROJECT, 'n_gpus': 8, 'sbatch_nodes': 2, 'gradient_accumulation_steps': 32, 'batch_size': 20, 'max_iters': 2465},
]

configs = []
for mup in [True]: #[True, False]:
    for lr in learing_rates:
        for wd in weight_decays:
            for seed in seeds:
                for conf in model_configs:
                    conf = deepcopy(conf)
                    conf['learning_rate'] = lr
                    conf['weight_decay'] = wd
                    conf['seed'] = seed

                    if mup:
                        conf['mup'] = 'true'
                        conf['mup_multiplier'] = conf['n_embd'] / 384
                        conf['impl'] = 'tpv_left_impl'
                    else:
                        conf['mup'] = 'false'
                        conf['mup_multiplier'] = 1
                        conf['impl'] = 'standard_param_impl'
                    configs.append(conf)

if __name__ == "__main__":
    import json
    import random

    # random.shuffle(configs)

    config_strings = []
    for config in configs:
        config_strings.append(json.dumps(config))

    print('\n'.join(config_strings))
