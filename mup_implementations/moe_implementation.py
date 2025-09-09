tpv_left_impl_new_kv_2 = {
    'name':                     'TPV-L, new KV (muP)',
    'embedding': {
        'init_std':             lambda m: 1.0,
        'lr_scale':             lambda m: 1.0,
        'wd_scale':             lambda m: 1.0,
        'output_multiplier':    lambda m: 1.0
    },
    'hidden': {
        'init_std':             lambda m: 1 / m**(1/2),
        'lr_scale':             lambda m: 1 / m,
        'wd_scale':             lambda m: m,
        'output_multiplier':    lambda m: 1.0
    },
    'kv_layer': {
        'init_std':             lambda m, r: 1 / m**(1/2),
        'lr_scale':             lambda m, r: (1 + r**(1/2)) / (2 * m),
        'wd_scale':             lambda m, r: 2 * m / (1 + r**(1/2)),
        'output_multiplier':    lambda m, r: 1.0
    },
    'unembedding': {
        'init_std':             lambda m: 1.0,
        'lr_scale':             lambda m: 1.0,
        'wd_scale':             lambda m: 1.0,
        'output_multiplier':    lambda m: 1 / m
    },
    'router': {
        'init_std':             lambda m, E: 1.0 / (m**(1/2) + E**(1/2)),
        'lr_scale':             lambda m, E: 1.0 / (m**(1/2) * E**(1/2)),
        'wd_scale':             lambda m, E: m,
        'output_multiplier':    lambda m, E: m
    },
    'experts': {
        'init_std':             lambda m, ffn_m: 1.0 / ffn_m,
        'lr_scale':             lambda m, ffn_m: 1.0 / ffn_m,
        'wd_scale':             lambda m, ffn_m: ffn_m,
        'output_multiplier':    lambda m, ffn_m: ffn_m
    },
    'normalization': {
        'lr_scale':             lambda m: 1.0,
    },
    'attention_scale':          lambda d: 1 / d,
    'depth_scale':              lambda L: 1 / L
}