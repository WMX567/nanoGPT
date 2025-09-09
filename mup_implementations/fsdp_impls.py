"""
    To enable good FSDP sharding we lose the ability to set per-layer
    learning rates. Thus, we can only adjust 
"""

# No MoE implemenation, KV scalings only
fsdp_full_impl = {
    'name':                     'FSDP full implementation',
    'lr_scale':                 lambda m: 1.0 / m,
    'wd_scale':                 lambda m: m,
    'attention_scale':          lambda d: 1 / d,
    'depth_scale':              lambda L: 1.0,

    'embedding': {
        'init_std':             lambda m: 1.0 / m,
        'output_multiplier':    lambda m: m,
    },

    'hidden': {
        'init_std':             lambda m: 1.0 / m**(1/2),
        'output_multiplier':    lambda m: 1.0,
    },

    'kv_layer': {
        'init_std':             lambda m, r: 2 / (m**(1/2) * r**(1/2) * (1 + r**(1/2))),
        'output_multiplier':    lambda m, r: r**(1/2) * (1 + r**(1/2)) / 2,
    },

    'unembedding': {
        'init_std':             lambda m: 1.0 / m,
        'output_multiplier':    lambda m: 1.0,
    },

    'router': {
        'init_std':             lambda m, E: 1.0 / m,
        'output_multiplier':    lambda m, E: m,
    },

    'experts': {
        'init_std':             lambda m, ffn_m: 1.0 / m,
        'output_multiplier':    lambda m, ffn_m: m,
    }
}