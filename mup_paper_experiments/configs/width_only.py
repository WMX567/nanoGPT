# Config for varying width only
import numpy as np
from copy import deepcopy

WEIGHT_DECAY = 0.0
MODEL_DEPTH = 8

LEARNING_RATE_SAMPLES = 1 # 30
learing_rates = [
    10**p for p in np.linspace(-4.25, -2.5, LEARNING_RATE_SAMPLES)
]
seeds = [42] #, 43, 44, 45, 46]

WANDB_PROJECT = 'width-only-sweep-big-depths'

model_configs = [
    # {'n_embd': 256, 'n_head': 4, 'n_kv_head': 4, 'n_layer': MODEL_DEPTH, 'weight_decay': 0.0, 'log_wandb': 'True', 'wandb_project': WANDB_PROJECT, 'n_gpus': 2, 'gradient_accumulation_steps': 2, 'batch_size': 50, 'max_iters': 392},
    # {'n_embd': 320, 'n_head': 5, 'n_kv_head': 5, 'n_layer': MODEL_DEPTH, 'weight_decay': 0.0, 'log_wandb': 'True', 'wandb_project': WANDB_PROJECT, 'n_gpus': 2, 'gradient_accumulation_steps': 2, 'batch_size': 58, 'max_iters': 456},
    # {'n_embd': 384, 'n_head': 6, 'n_kv_head': 6, 'n_layer': MODEL_DEPTH, 'weight_decay': 0.0, 'log_wandb': 'True', 'wandb_project': WANDB_PROJECT, 'n_gpus': 2, 'gradient_accumulation_steps': 2, 'batch_size': 66, 'max_iters': 518},
    # {'n_embd': 448, 'n_head': 7, 'n_kv_head': 7, 'n_layer': MODEL_DEPTH, 'weight_decay': 0.0, 'log_wandb': 'True', 'wandb_project': WANDB_PROJECT, 'n_gpus': 2, 'gradient_accumulation_steps': 2, 'batch_size': 74, 'max_iters': 579},
    # {'n_embd': 512, 'n_head': 8, 'n_kv_head': 8, 'n_layer': MODEL_DEPTH, 'weight_decay': 0.0, 'log_wandb': 'True', 'wandb_project': WANDB_PROJECT, 'n_gpus': 2, 'gradient_accumulation_steps': 2, 'batch_size': 81, 'max_iters': 639},
    {'n_embd': 1024, 'n_head': 16,  'n_kv_head': 16,  'n_layer': MODEL_DEPTH, 'weight_decay': 0.0, 'log_wandb': 'true', 'wandb_project': WANDB_PROJECT, 'n_gpus': 4, 'gradient_accumulation_steps': 4, 'batch_size': 70, 'max_iters': 1104},
    {'n_embd': 2048, 'n_head': 32,  'n_kv_head': 32,  'n_layer': MODEL_DEPTH, 'weight_decay': 0.0, 'log_wandb': 'true', 'wandb_project': WANDB_PROJECT, 'n_gpus': 8, 'gradient_accumulation_steps': 8, 'batch_size': 64, 'max_iters': 2013},
]

configs = []
for mup in [True]:#, False]:
    for lr in learing_rates:
        for seed in seeds:
            for conf in model_configs:
                conf = deepcopy(conf)
                conf['learning_rate'] = lr
                conf['seed'] = seed
                if mup:
                    conf['mup'] = 'true'
                    conf['mup_multiplier'] = conf['n_embd'] / 256
                    conf['impl'] = 'tpv_left_impl'
                else:
                    conf['mup'] = 'false'
                    conf['mup_multiplier'] = 1
                    conf['impl'] = 'standard_param_impl'
                configs.append(conf)

if __name__ == "__main__":
    import json
    import random

    random.shuffle(configs)

    config_strings = []
    for config in configs:
        config_strings.append(json.dumps(config))

    print('\n'.join(config_strings))

# Generate the json for this

