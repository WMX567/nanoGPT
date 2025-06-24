# TODO: This is not the correct standard param implementation.
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
        'init_std':             lambda m: 1.0,
        'lr_scale':             lambda m: 1.0,
        'wd_scale':             lambda m: 1.0,
        'output_multiplier':    lambda m: 1.0
    },
    'normalization': {
        'lr_scale':             lambda m: 1.0,
    },
    'attention_scale':          lambda d: 1 / d**(1/2)
}

tpv_left_impl = {
    'name':                     'TPV-L (muP)',
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
    'unembedding': {
        'init_std':             lambda m: 1.0,
        'lr_scale':             lambda m: 1.0,
        'wd_scale':             lambda m: 1.0,
        'output_multiplier':    lambda m: 1 / m
    },
    'normalization': {
        'lr_scale':             lambda m: 1.0,
    },
    'attention_scale':          lambda d: 1 / d
}

# Table from Cerebras which fixed the learning rate across
# all of the layers.
tpv_right_impl = {
    'name':                     'TPV-R (muP)',
    'embedding': {
        'init_std':             lambda m: 1.0,
        'lr_scale':             lambda m: 1.0,
        'wd_scale':             lambda m: 1.0,
        'output_multiplier':    lambda m: 1.0
    },
    'hidden': {
        'init_std':             lambda m: m**(1/2),
        'lr_scale':             lambda m: 1,
        'wd_scale':             lambda m: 1,
        'output_multiplier':    lambda m: 1 / m
    },
    'unembedding': {
        'init_std':             lambda m: 1.0,
        'lr_scale':             lambda m: 1.0,
        'wd_scale':             lambda m: 1.0,
        'output_multiplier':    lambda m: 1 / m
    },
    'normalization': {
        'lr_scale':             lambda m: 1.0,
    },
    'attention_scale':          lambda d: 1 / d
}

# Untied weights.
# Table from IFM which fixes learning rate and ensure that
# the outputs land in bf16 range.
xllm_impl = {
    'name':                     'xLLM (muP)',
    'embedding': {
        'init_std':             lambda m: 1.0 / m,
        'lr_scale':             lambda m: 1.0 / m,
        'wd_scale':             lambda m: m,
        'output_multiplier':    lambda m: m,
    },
    'hidden': {
        'init_std':             lambda m: 1.0 / m**(1/2),
        'lr_scale':             lambda m: 1.0 / m,
        'wd_scale':             lambda m: m,
        'output_multiplier':    lambda m: 1.0
    },
    'unembedding': {
        'init_std':             lambda m: 1.0 / m,
        'lr_scale':             lambda m: 1.0 / m,
        'wd_scale':             lambda m: m,
        'output_multiplier':    lambda m: 1.0 
    },
    'normalization': {
        'lr_scale':             lambda m: 1.0 / m,
    },
    'attention_scale':          lambda d: 1 / d
}