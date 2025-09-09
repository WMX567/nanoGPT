# KV and router weights are 1.0 by default.
standard_param_impl = {
    'name':                     'SP',
    'embedding': {
        'init_std':             lambda m: 1.0,
        'lr_scale':             lambda m: 1.0,
        'wd_scale':             lambda m: 1.0,
        'output_multiplier':    lambda m: 1.0
    },
    'hidden': {
        'init_std':             lambda m: 1.0 / m**(1/2),
        'lr_scale':             lambda m: 1.0,
        'wd_scale':             lambda m: 1.0,
        'output_multiplier':    lambda m: 1.0
    },
    'unembedding': {
        'init_std':             lambda m: 1.0 / m**(1/2),
        'lr_scale':             lambda m: 1.0,
        'wd_scale':             lambda m: 1.0,
        'output_multiplier':    lambda m: 1.0
    },
    'normalization': {
        'lr_scale':             lambda m: 1.0,
    },
    'attention_scale':          lambda d: 1 / d**(1/2),
    'depth_scale':              lambda L: 1.0,
}

standard_param_impl_completep_depth_scaling = {
    'name':                     'SP with Complete-P depth scaling',
    'embedding': {
        'init_std':             lambda m: 1.0,
        'lr_scale':             lambda m: 1.0,
        'wd_scale':             lambda m: 1.0,
        'output_multiplier':    lambda m: 1.0
    },
    'hidden': {
        'init_std':             lambda m: 1.0 / m**(1/2),
        'lr_scale':             lambda m: 1.0,
        'wd_scale':             lambda m: 1.0,
        'output_multiplier':    lambda m: 1.0
    },
    'unembedding': {
        'init_std':             lambda m: 1.0 / m**(1/2),
        'lr_scale':             lambda m: 1.0,
        'wd_scale':             lambda m: 1.0,
        'output_multiplier':    lambda m: 1.0
    },
    'normalization': {
        'lr_scale':             lambda m: 1.0,
    },
    'attention_scale':          lambda d: 1 / d**(1/2),
    'depth_scale':              lambda L: 1 / L,
}

standard_param_impl_tpvi_depth_scaling = {
    'name':                     'SP with TP6 depth scaling',
    'embedding': {
        'init_std':             lambda m: 1.0,
        'lr_scale':             lambda m: 1.0,
        'wd_scale':             lambda m: 1.0,
        'output_multiplier':    lambda m: 1.0
    },
    'hidden': {
        'init_std':             lambda m: 1.0 / m**(1/2),
        'lr_scale':             lambda m: 1.0,
        'wd_scale':             lambda m: 1.0,
        'output_multiplier':    lambda m: 1.0
    },
    'unembedding': {
        'init_std':             lambda m: 1.0 / m**(1/2),
        'lr_scale':             lambda m: 1.0,
        'wd_scale':             lambda m: 1.0,
        'output_multiplier':    lambda m: 1.0
    },
    'normalization': {
        'lr_scale':             lambda m: 1.0,
    },
    'attention_scale':          lambda d: 1 / d**(1/2),
    'depth_scale':              lambda L: 1 / L,
}