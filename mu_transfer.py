
import torch
import pandas as pd
import os
import numpy as np
import pickle
import random
import torch
import sys

from contextlib import nullcontext
from megatron.core.datasets.indexed_dataset import IndexedDataset

from model_moe_kyle import GPTConfig, GPT
from mup_implementations import (
    impl_dict
)

# 确保输出能够及时刷新
sys.stdout.flush()
sys.stderr.flush()

def build_model(width, param_type, device):
    # param_type: 'sp' or 'mup'
    if param_type == 'sp':
        mup_flag = False
        impl = impl_dict['standard_param_impl']
        mup_multiplier = 1.0
    else:
        mup_flag = True
        impl = impl_dict['mengxi_impl']
        mup_multiplier = width / 512

    model_args = dict(
        n_layer=n_layer, n_head=n_head, n_kv_head=n_kv_head, n_embd=width, block_size=block_size,
        bias=bias, vocab_size=meta_vocab_size if 'meta_vocab_size' in globals() and meta_vocab_size is not None else 50304, dropout=dropout,
        mup=mup_flag, mup_multiplier=mup_multiplier, init_std=init_std, impl=impl, normalization=normalization,
        use_moe=use_moe, num_experts=num_experts, router_topk=router_topk, moe_ffn_hidden_size=moe_ffn_hidden_size,
        moe_seq_aux_loss_coeff=moe_seq_aux_loss_coeff, moe_ffn_mup_multiplier=moe_ffn_mup_multiplier,
        moe_random_router=moe_random_router
    )
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf).to(device)
    return model

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
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
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
warmup_iters = 2000 # how many steps to warm up for
weight_decay = 0.0
beta1 = 0.9
beta2 = 0.95
eps = 1e-8 
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 0 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
# dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
dtype = 'float32'
compile = True # use PyTorch 2.0 to compile the model to be faster
mup = True
mup_multiplier = 1.0
coord_check = True
# impl = tpv_left_impl
impl = 'mengxi_impl'
seed = 42
bias = False
init_std = 0.02
coord_check = True
normalization = "LayerNorm"
tag = "default"  # Tag for coordinate checking data
use_moe: bool = False
num_experts: int = 0
router_topk: int = 8
moe_ffn_hidden_size: int = 128
moe_seq_aux_loss_coeff: float = 1e-4
moe_ffn_mup_multiplier: float = 1.0
moe_random_router: bool = False
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('/scratch1/mengxiwu/nanoGPT/configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

os.makedirs(out_dir, exist_ok=True)
seed_everything(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ptimpl = impl_dict[impl] if isinstance(impl, str) else impl
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

if dataset != 'slim_pajama':
    # poor man's data loader
    if dataset == 'openwebtext':
        data_dir = '/scratch1/mengxiwu/nanoGPT/data/openwebtext'
    else:
        data_dir = os.path.join('data', dataset)
    def get_batch(split):
        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        if split == 'train':
            if dataset == 'slim_pajama':
                data = np.memmap('/mnt/sharefs/data/pretrain_tokenized/SlimPajama-627B_250k_tokenized/merged/slimpajama-train-chunk1.bin', dtype=np.uint16, mode='r')
            else:
                data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        # Device will be set in the main function
        return x, y
else:
    path_prefix = "/mnt/sharefs/data/pretrain_tokenized/SlimPajama-627B_250k_tokenized/merged/slimpajama-train-chunk1"
    slimpj_dataset = IndexedDataset(path_prefix)

    data_buffer = []
    dataset_idx = 0

    def get_batch(_):
        global dataset_idx, data_buffer

        X_batch = []
        Y_batch = []

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

        x = torch.tensor(X_batch, dtype=torch.long)
        y = torch.tensor(Y_batch, dtype=torch.long)

        # Device will be set in the main function
        return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

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
    meta_vocab_size = 250221

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_kv_head=n_kv_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout, mup=mup, 
                  mup_multiplier=mup_multiplier, init_std=init_std,
                  impl=ptimpl, normalization=normalization, use_moe=use_moe, 
                  num_experts=num_experts, router_topk=router_topk, 
                  moe_ffn_hidden_size=moe_ffn_hidden_size, 
                  moe_seq_aux_loss_coeff=moe_seq_aux_loss_coeff, 
                  moe_ffn_mup_multiplier=moe_ffn_mup_multiplier,
                  moe_random_router=moe_random_router) # start with model_args from command line

# init a new model from scratch
print("Initializing a new model from scratch")
# determine the vocab size we'll use for from-scratch training
if meta_vocab_size is None:
    print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
print("MODEL ARGS:", model_args)
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)



import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def main_worker(local_rank, world_size):
    def print_mfu(model, fwdbwd_per_iter, dt):
        try:
            mfu = model.module.estimate_mfu(fwdbwd_per_iter, dt)
            print(f"[GPU {local_rank}] MFU: {mfu:.4f}")
        except Exception as e:
            print(f"[GPU {local_rank}] MFU error: {e}")
    
    # Initialize distributed training
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=local_rank)
    torch.cuda.set_device(local_rank)
    device = f'cuda:{local_rank}'
    seed_everything(seed + local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    param_type = 'mup'
    model = build_model(n_embd, param_type, device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = model.module.configure_optimizers(
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        betas=(beta1, beta2),
        eps=eps,
        device_type=device_type,
        adaptive_optimizer=False
    )
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))
    losses = []
    optimizer.zero_grad(set_to_none=True)
    import time
    t0 = time.time()
    for step in range(max_iters):
        x, y = get_batch('train')
        with ctx:
            outputs = model(x, y)
            if isinstance(outputs, tuple):
                loss = outputs[1] if outputs[1] is not None else outputs[0]
            else:
                loss = outputs
            loss = loss / gradient_accumulation_steps
        scaler.scale(loss).backward()
        if (step + 1) % gradient_accumulation_steps == 0:
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item() * gradient_accumulation_steps)
        # Print loss and MFU every 10 steps (only on rank 0)
        if (step + 1) % 10 == 0 and local_rank == 0:
            dt = time.time() - t0
            print(f"[GPU {local_rank}] Step {step+1} | Loss: {loss.item() * gradient_accumulation_steps:.6f}")
            print_mfu(model, gradient_accumulation_steps, dt)
            t0 = time.time()
    if local_rank == 0:
        out_dir = f"mu_transfer_results/{param_type}/"
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame({
            'step': np.arange(max_iters),
            'loss': losses,
            'width': n_embd,
            'log2lr': np.log2(learning_rate),
            'param_type': param_type
        }).to_csv(f"{out_dir}/width{n_embd}_log2lr{np.log2(learning_rate):.2f}.csv", index=False)


def main():
    print("=== MU TRANSFER TRAINING STARTING ===")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
    sys.stdout.flush()

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # We're running under torchrun, use the existing distributed setup
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        global_rank = int(os.environ['RANK'])
        
        # Initialize distributed training
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = f'cuda:{local_rank}'
        use_ddp = True
        
        def print_mfu(model, fwdbwd_per_iter, dt):
            try:
                mfu = model.module.estimate_mfu(fwdbwd_per_iter, dt)
                if global_rank == 0:
                    print(f"[GPU {local_rank}] MFU: {mfu:.4f}")
            except Exception as e:
                if global_rank == 0:
                    print(f"[GPU {local_rank}] MFU error: {e}")
        
        # Set up device and context
        seed_everything(seed + global_rank)
    else:
        # Single GPU training
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            torch.cuda.set_device(0)
        use_ddp = False
        global_rank = 0
        local_rank = 0
        
        def print_mfu(model, fwdbwd_per_iter, dt):
            try:
                if hasattr(model, 'module'):
                    mfu = model.module.estimate_mfu(fwdbwd_per_iter, dt)
                else:
                    mfu = model.estimate_mfu(fwdbwd_per_iter, dt)
                print(f"MFU: {mfu:.4f}")
            except Exception as e:
                print(f"MFU error: {e}")
        
        # Set up device and context
        seed_everything(seed)
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    param_type = 'mup'
    print(f"Building model with n_embd={n_embd}, param_type={param_type}")
    sys.stdout.flush()
    model = build_model(n_embd, param_type, device)
    print(f"Model built successfully. Device: {device}")
    sys.stdout.flush()
    
    if use_ddp:
        print("Wrapping model with DDP")
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        print("DDP wrapping completed")
        sys.stdout.flush()
    
    model_for_optimizer = model.module if use_ddp else model
    print("Configuring optimizer...")
    sys.stdout.flush()
    optimizer = model_for_optimizer.configure_optimizers(
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        betas=(beta1, beta2),
        eps=eps,
        device_type=device_type,
        adaptive_optimizer=False
    )
    print("Optimizer configured successfully")
    sys.stdout.flush()
    
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))
    losses = []
    optimizer.zero_grad(set_to_none=True)
    
    print(f"Starting training loop for {max_iters} iterations...")
    print(f"Batch size: {batch_size}, Gradient accumulation steps: {gradient_accumulation_steps}")
    sys.stdout.flush()
    
    import time
    t0 = time.time()
    for step in range(max_iters):
        x, y = get_batch('train')
   
        if device_type == 'cuda':
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        
        with ctx:
            outputs = model(x, y)
            if isinstance(outputs, tuple):
                loss = outputs[1] if outputs[1] is not None else outputs[0]
            else:
                loss = outputs
            loss = loss / gradient_accumulation_steps
        scaler.scale(loss).backward()
        if (step + 1) % gradient_accumulation_steps == 0:
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item() * gradient_accumulation_steps)

        # Print loss every 5 steps instead of 10 for more frequent output
        if (step + 1) % 5 == 0 and global_rank == 0:
            dt = time.time() - t0
            current_loss = loss.item() * gradient_accumulation_steps
            print(f"Step {step+1}/{max_iters} | Loss: {current_loss:.6f} | Time: {dt:.2f}s")
            sys.stdout.flush()
            print_mfu(model, gradient_accumulation_steps, dt)
            t0 = time.time()
        
        # Also print every 50 steps for less frequent but more detailed info
        if (step + 1) % 50 == 0 and global_rank == 0:
            avg_loss = np.mean(losses[-50:]) if len(losses) >= 50 else np.mean(losses)
            print(f"=== Step {step+1} Summary ===")
            print(f"Current loss: {current_loss:.6f}")
            print(f"Average loss (last 50): {avg_loss:.6f}")
            print(f"Progress: {(step+1)/max_iters*100:.1f}%")
            sys.stdout.flush()
    
    if global_rank == 0:
        print("Training completed! Saving results...")
        sys.stdout.flush()
        out_dir_full = f"{out_dir}/{param_type}/"
        os.makedirs(out_dir_full, exist_ok=True)
        
        results_df = pd.DataFrame({
            'step': np.arange(max_iters),
            'loss': losses,
            'width': n_embd,
            'log2lr': np.log2(learning_rate),
            'param_type': param_type
        })
        
        csv_path = f"{out_dir_full}/width{n_embd}_log2lr{np.log2(learning_rate):.2f}.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"Results saved to: {csv_path}")
        
        # 也保存一个简单的总结
        summary_path = f"{out_dir_full}/training_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Training Summary\n")
            f.write(f"===============\n")
            f.write(f"Model width: {n_embd}\n")
            f.write(f"Learning rate: {learning_rate}\n")
            f.write(f"Weight decay: {weight_decay}\n")
            f.write(f"Batch size: {batch_size}\n")
            f.write(f"Gradient accumulation steps: {gradient_accumulation_steps}\n")
            f.write(f"Total steps: {max_iters}\n")
            f.write(f"Final loss: {losses[-1]:.6f}\n")
            f.write(f"Average loss: {np.mean(losses):.6f}\n")
            f.write(f"Min loss: {np.min(losses):.6f}\n")
        
        print(f"Training summary saved to: {summary_path}")
        sys.stdout.flush()
    
    if use_ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()