# Config for varying width only

import numpy as np

WEIGHT_DECAY = 0.0
HEAD_SIZE = 64
MODEL_DEPTH = 6
WIDTHS = [
    256, 320, 384
]

LEARNING_RATE_SAMPLES = 30
learing_rates = [
    10**p for p in np.linspace(-4.25, -2.5, LEARNING_RATE_SAMPLES)
]

configs = []
for lr in learing_rates:
    for width in WIDTHS:
        config = {
            'n_embd': width,
            'n_head': width // HEAD_SIZE,
            'n_kv_head': width // HEAD_SIZE,
            'n_layer': MODEL_DEPTH,
            'weight_decay': WEIGHT_DECAY,
            'learning_rate': lr,
            'n_gpus': 2,
            'gradient_accumulation_steps': 2,
            'wandb_project': 'slurm-shakedown',
            'max_iters': 500,
            'log_wandb': 'True',
        }
        configs.append(config)

if __name__ == "__main__":
    import json

    config_strings = []
    for config in configs:
        config_strings.append(json.dumps(config))

    print('\n'.join(config_strings))
