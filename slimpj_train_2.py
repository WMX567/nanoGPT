"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import math
import numpy as np
import os
import pickle
import random
import time
import torch

from contextlib import nullcontext
from megatron.core.datasets.indexed_dataset import _IndexReader, IndexedDataset
from torch.distributed import destroy_process_group, init_process_group
from torch.distributed.fsdp import fully_shard, FSDPModule
from torch.nn.parallel import DistributedDataParallel as DDP

# from model import GPTConfig, GPT
from model_moe_kyle import GPTConfig, GPT
from mup_implementations import (
    impl_dict
)

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_kv_head = n_head
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
adaptive_optimizer = False
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 0.0
beta1 = 0.9
beta2 = 0.95
eps = 1e-12
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
cooldown_iters = 2000
decay_profile = 'cosine' # ['cosine', 'wsd', 'wsd_cosine_tail']
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
# dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
dtype = 'float32'
mup = True
mup_multiplier = 1.0
# impl = tpv_left_impl
impl = 'xllm_impl'
seed = 42
bias = False
init_std = 0.02
coord_check = False
avg_interval = 30
normalization = "LayerNorm"
# prelayer norm options ['None', 'LayerNorm', 'L2Norm']
q_prelayer_normalization = 'None'
k_prelayer_normalization = 'None'
complete_p_layers = False
slurm_job_id: int = 0
slurm_array_task_id: int =  0
anneal_wd = False
min_wd = 0.0 # minimum weight decay for annealing
wd_warmup_iters = 1000
wd_anneal_iters = 1000
# -------- MoE-related knobs ----------
use_moe: bool = False
num_experts: int = 0
router_topk: int = 8
moe_ffn_hidden_size: int = 128
moe_seq_aux_loss_coeff: float = 1e-4
moe_ffn_mup_multiplier: float = 1.0
moe_null_expert_bias: float = 0.0
moe_random_router: bool = False
# -------- FSDP ----------
enable_fsdp = False
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('/scratch1/mengxiwu/nanoGPT/configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

assert not (enable_fsdp and compile), "FSDP and PyTorch 2.0 compilation are mutually exclusive, please disable one of them."

os.makedirs(out_dir, exist_ok=True)
seed_everything(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ptimpl = impl_dict[impl] if isinstance(impl, str) else impl
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size

    print(f"running in DDP mode with {ddp_world_size} processes, local rank {ddp_local_rank}, device {device}")
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
# torch.manual_seed(1337 + seed_offset)
seed_everything(seed + seed_offset) # set the random seed for reproducibility
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
if dataset != 'slim_pajama':
    data_dir = os.path.join('data', dataset)
    def get_batch(split):
        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        if split == 'train':
            data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        if device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y
else:
    slim_pj_cropped_vocab_size = 50257
    slim_pj_full_vocab_size = 250222

    compression_ratio = slim_pj_full_vocab_size // slim_pj_cropped_vocab_size

    path_prefix = "/mnt/sharefs/data/pretrain_tokenized/SlimPajama-627B_250k_tokenized/merged/slimpajama-train-chunk1"
    slimpj_dataset = IndexedDataset(path_prefix)

    dataset_idx = 0

    def get_batch(_):
        global dataset_idx

        X_batch = []
        Y_batch = []

        data_buffer = []

        while len(X_batch) < batch_size:
            while len(data_buffer) < (2 * block_size):
                new_sample = slimpj_dataset.get(dataset_idx)
                dataset_idx += 1

                if new_sample is None:
                    if not X_batch:
                        raise StopIteration("End of dataset reached and no full batches could be formed.")
                    else:
                        break
                
                data_buffer.extend(new_sample)

            while len(data_buffer) >= (block_size + 1) and len(X_batch) < batch_size:
                x = data_buffer[:block_size]
                y = data_buffer[1:block_size+1]

                X_batch.append(x)
                Y_batch.append(y)

                data_buffer = data_buffer[1:]

        # x = torch.tensor(X_batch, dtype=torch.long)
        # y = torch.tensor(Y_batch, dtype=torch.long)

        x = torch.tensor(X_batch) / compression_ratio
        y = torch.tensor(Y_batch) / compression_ratio

        x = x.to(dtype=torch.long)
        y = y.to(dtype=torch.long)

        x = x.clip(0, slim_pj_cropped_vocab_size - 1)
        y = y.clip(0, slim_pj_cropped_vocab_size - 1)

        # print(f"shapes: {x.shape}, {y.shape}")

        # x = torch.div(x, compression_ratio, rounding_mode='floor')
        # y = torch.div(y, compression_ratio, rounding_mode='floor')

        if device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        
        del data_buffer
        
        return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

avg_idx = 0
avg_loss = [0.0] * avg_interval  # Initialize avg_loss with zeros

# attempt to derive vocab_size from the dataset
if dataset != 'slim_pajama':
    meta_path = os.path.join(data_dir, 'meta.pkl')
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta['vocab_size']
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
else:
    meta_vocab_size = slim_pj_cropped_vocab_size

# model init
# model init
model_args = dict(
    n_layer=n_layer, 
    n_head=n_head, 
    n_kv_head=n_kv_head, 
    n_embd=n_embd, 
    block_size=block_size,
    bias=bias, 
    vocab_size=None, 
    dropout=dropout, 
    mup=mup, 
    mup_multiplier=mup_multiplier, 
    init_std=init_std,
    impl=ptimpl, 
    normalization=normalization, 
    complete_p_layers=complete_p_layers,
    q_prelayer_normalization=q_prelayer_normalization, 
    k_prelayer_normalization=k_prelayer_normalization,
    use_moe=use_moe,
    num_experts=num_experts,
    moe_ffn_hidden_size=moe_ffn_hidden_size,
    router_topk=router_topk,
    moe_seq_aux_loss_coeff=moe_seq_aux_loss_coeff,
    moe_ffn_mup_multiplier=moe_ffn_mup_multiplier,
    moe_null_expert_bias=moe_null_expert_bias,
    moe_random_router=moe_random_router,
)

if init_from == 'scratch':
    print("Initializing a new model from scratch")
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    print("MODEL ARGS:", model_args)
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    model_params = model.get_num_params(non_embedding=False) / 1e6
    non_embedding_params = model.get_num_params(non_embedding=True) / 1e6

    if coord_check:
        from coordinate_checking import get_hooks
        data = {}
        hooks = []
        for i, layer in enumerate(model.transformer.h):
            mlp = layer.mlp

            fc = mlp.c_fc
            forward_hook = get_hooks(data, f"fc.{i}")
            hook = fc.register_forward_hook(forward_hook)
            hooks.append(hook)

            c_proj = mlp.c_proj
            forward_hook = get_hooks(data, f"c_proj.{i}")
            hook = c_proj.register_forward_hook(forward_hook)
            hooks.append(hook)

            attn = layer.attn
            forward_hook = get_hooks(data, f"attn.c_attn.{i}")
            hook = attn.c_attn.register_forward_hook(forward_hook)
            hooks.append(hook)

            forward_hook = get_hooks(data, f"attn.c_proj.{i}")
            hook = attn.c_proj.register_forward_hook(forward_hook)
            hooks.append(hook)

        forward_hook = get_hooks(data, f"lm_head")
        hook = model.lm_head.register_forward_hook(forward_hook)
        hooks.append(hook)
        
        # forward_hook = get_hooks(data, "output")
        # hook = model.output_id.register_forward_hook(forward_hook)
        # hooks.append(hook)
        

elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    # MoE can leave some expert parameters unused on a given forward pass which
    # breaks the default DDP reduction. Enable find_unused_parameters so DDP
    # tolerates parameters that do not participate in every forward.
    if enable_fsdp: 
        for layer in model.transformer.h:
            try:
                fully_shard(layer)
            except Exception as e:
                print(f"Failed to fully shard layer {layer}: {e}")
        fully_shard(model)
        assert isinstance(model, FSDPModule)
    else:
        model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=False)

# optimizer
if ddp and not enable_fsdp:
    raw_model = model.module # unwrap DDP
else:
    raw_model = model

optimizer = raw_model.configure_optimizers(
    weight_decay, learning_rate, (beta1, beta2), eps, device_type,
    adaptive_optimizer=adaptive_optimizer,
)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                if use_moe:
                    logits, loss, _, _ = model(X, Y)
                else:
                    logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr_wsd(it):
    # Warm up, warm down, tail type (cosine, linear), min_lr
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    elif warmup_iters <= it and it < lr_decay_iters - cooldown_iters:
        return learning_rate
    elif it >= lr_decay_iters - cooldown_iters and it < lr_decay_iters:
        decay_ratio = (it - (lr_decay_iters - cooldown_iters)) / cooldown_iters
        if decay_profile == 'wsd_cosine_tail':
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return min_lr + coeff * (learning_rate - min_lr)
        else:
            assert 0 <= decay_ratio <= 1
            return learning_rate - decay_ratio * (learning_rate - min_lr) 
    else:
        return min_lr

def get_lr_cosine(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

def get_lr_const(it):
    return learning_rate

def get_wd_annealed(it):
    if it < wd_warmup_iters:
        return weight_decay
    elif it < wd_warmup_iters + wd_anneal_iters:
        decay_ratio = (it - wd_warmup_iters) / wd_anneal_iters
        assert 0 <= decay_ratio <= 1
        return weight_decay - (weight_decay - min_wd) * decay_ratio
    else:
        return min_wd

def get_wd_const(it):
    return weight_decay


if coord_check:
    get_lr = get_lr_const
    get_wd = get_wd_const

if anneal_wd:
    get_wd = get_wd_annealed
else:
    get_wd = get_wd_const

if decay_profile == 'wsd' or decay_profile == 'wsd_cosine_tail':
    assert cooldown_iters is not None, "cooldown_iters must be set for WSD decay profile"
    get_lr = get_lr_wsd
elif decay_profile == 'cosine':
    get_lr = get_lr_cosine
else:
    raise ValueError(f"Unknown decay profile: {decay_profile}. Supported: ['wsd', 'cosine']")
    

config['model_params'] = model_params
config['non_embedding_params'] = non_embedding_params
# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

tiwd = 0 # total integrated weight decay
tilr = 0 # total integrated learning rate
# training loop
X, Y = get_batch('train') # fetch the very first batch
global_time = time.time()
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp and not enable_fsdp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    wd = get_wd(iter_num) if anneal_wd else weight_decay
    tiwd += lr * wd
    tilr += lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group.get('lr_scale', 1.0)
        param_group['weight_decay'] = wd * param_group.get('wd_scale', 1.0)

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if master_process:
            if wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "wd": wd,
                    "tiwd": tiwd,
                    "tilr": tilr,
                    "base_lr": learning_rate,
                    "mfu": running_mfu*100, # convert to percentage
                })
        # TODO: I've disabled checkpointing for the moment.
        # if losses['val'] < best_val_loss or always_save_checkpoint:
        #     best_val_loss = losses['val']
        #     if iter_num > 0:
        #         checkpoint = {
        #             'model': raw_model.state_dict(),
        #             'optimizer': optimizer.state_dict(),
        #             'model_args': model_args,
        #             'iter_num': iter_num,
        #             'best_val_loss': best_val_loss,
        #             'config': config,
        #         }
        #         if master_process:
        #             print(f"saving checkpoint to {out_dir}")
        #             torch.save(checkpoint, os.path.join(out_dir, f'ckpt_{ckpt_num}.pt'))
        #             print(f"saved checkpoint to {out_dir}/ckpt_{ckpt_num}.pt")
        #             ckpt_num += 1

        # # sync all ranks after checkpoint logic
        # if ddp:
        #     torch.distributed.barrier()
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    # accumulate MoE aux loss across micro-steps to log properly
    if use_moe:
        moe_aux_accum = torch.tensor(0.0, device=device)
    else:
        moe_aux_accum = None

    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            if use_moe:
                logits, loss, moe_aux, lm_loss = model(X, Y)
                # detach and accumulate the raw aux (sum over micro-steps). Do not .item() here
                # moe_aux_accum = moe_aux_accum + moe_aux.detach().to(device)
                moe_aux = moe_aux.detach() / gradient_accumulation_steps
                lm_loss = lm_loss.detach() / gradient_accumulation_steps
            else:
                logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()

    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()

    # abs_grad_norm = model.transformer.h[-1].attn.c_kv.weight.grad.abs().mean().item()

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if use_moe:
            moe_auxf = moe_aux * gradient_accumulation_steps
            lm_lossf = lm_loss * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

        avg_loss[avg_idx] = lossf
        avg_idx = (avg_idx + 1) % avg_interval

        if wandb_log:
            # Iterate through all the weights and get the F norm of them
            layer_norm_data = {}
            if not enable_fsdp:
                # TODO: Does this break things when we use FSDP?
                # I.e. no longer iterate through named params?
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        layer_norm_data[f"grad_norm/{name}"] = param.grad.norm().item()
                        layer_norm_data[f"norm/{name}"] = param.norm().item()

            moe_data = {}
            if use_moe:
                moe_data = {
                    "moe_aux_loss": moe_auxf,
                    "train/lm_loss": lm_lossf,
                }

            perf_data = {}
            if local_iter_num >= 4: # let the training loop settle a bit
                tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
                perf_data = {
                    "perf/throughput": tokens_per_iter / dt if dt > 0 else 0,
                    "perf/mfu": running_mfu * 100, # convert to percentage
                    "perf/iters_per_sec": 1 / dt if dt > 0 else 0,
                    "perf/average_iters_per_sec": iter_num / (time.time() - global_time) if iter_num > 0 else 0,
                }

            wandb.log({
                "iter": iter_num,
                "train/loss": lossf,
                "lr": lr,
                "tiwd": tiwd,
                "tilr": tilr,
                "wd": wd,
                "train/avg_loss": sum(avg_loss) / len(avg_loss),
                **layer_norm_data,
                **moe_data,
                **perf_data,
                # "abs_grad_norm": abs_grad_norm,
            })

    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if coord_check:
    import datetime
    from coordinate_checking import dataframe_from_data
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    df = dataframe_from_data(data, width=config['n_embd'], seed=seed)
    df.to_csv(os.path.join(out_dir, f'coord_check_{now}.csv'), index=False)
    print(f"Attempted to save coordinate checking data to {os.path.join(out_dir, f'coord_check_{now}.csv')}")

# Save losses!


if ddp:
    destroy_process_group()
