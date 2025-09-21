import os
import random
import pickle
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import nullcontext
from model_moe_kyle import GPTConfig, GPT
from mup_implementations import impl_dict


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) # 如果使用多GPU，应为 torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_args():
    parser = argparse.ArgumentParser(description='Train a GPT model with muP.')
    parser.add_argument('--out_dir', type=str, default='out', help='Output directory for checkpoints and logs.')
    parser.add_argument('--dataset', type=str, default='openwebtext', help='Dataset name (e.g., openwebtext).')
    parser.add_argument('--n_layer', type=int, default=12, help='Number of transformer layers.')
    parser.add_argument('--n_head', type=int, default=12, help='Number of attention heads.')
    parser.add_argument('--n_kv_head', type=int, default=12, help='Number of key/value heads (for GQA).')
    parser.add_argument('--n_embd', type=int, default=768, help='Embedding dimension (model width).')
    parser.add_argument('--block_size', type=int, default=1024, help='Sequence length.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
    parser.add_argument('--bias', type=bool, default=False, help='Use bias in Linear and LayerNorm layers.')
    parser.add_argument('--init_std', type=float, default=0.02, help='Initialization standard deviation for muP.')

    parser.add_argument('--batch_size', type=int, default=12, help='Micro-batch size.')
    parser.add_argument('--max_iters', type=int, default=600000, help='Total number of training iterations.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=40, help='Simulate larger batch sizes.')
    
    parser.add_argument('--learning_rate', type=float, default=6e-4, help='Max learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay.')
    parser.add_argument('--beta1', type=float, default=0.9, help='AdamW beta1.')
    parser.add_argument('--beta2', type=float, default=0.95, help='AdamW beta2.')
    parser.add_argument('--eps', type=float, default=1e-8, help='AdamW epsilon.')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value.')
    parser.add_argument('--decay_lr', type=bool, default=False, help='Disable learning rate decay.')

    parser.add_argument('--device', type=str, default='cuda', help='Device to use (e.g., cpu, cuda, cuda:0).')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16', 'float16'], help='Data type for training.')
    parser.add_argument('--compile', type=bool, default=False, help='Disable PyTorch 2.0 model compilation.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    
    parser.add_argument('--use_moe', action='store_true', help='Enable Mixture of Experts.')
    parser.add_argument('--num_experts', type=int, default=8, help='Number of experts in MoE.')
    parser.add_argument('--router_topk', type=int, default=2, help='Number of experts to route to for each token.')

    return parser.parse_args()


data_loaders = {'train': None, 'val': None}

def init_data_loader(split, data_dir, block_size):
    
    global data_loaders
    if data_loaders.get(split) is None:
        file_path = os.path.join(data_dir, f'{split}.bin')
        if os.path.exists(file_path):
            data_loaders[split] = np.memmap(file_path, dtype=np.uint16, mode='r')
        else:
            raise FileNotFoundError(f"Data file not found at {file_path}")

def get_batch(split, batch_size, block_size, device):

    data = data_loaders.get(split)
    if data is None:
        raise RuntimeError(f"Data loader for split '{split}' not initialized.")
        
    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.empty(batch_size, block_size, dtype=torch.long)
    y = torch.empty(batch_size, block_size, dtype=torch.long)
    
    for j, i in enumerate(ix):
        chunk_x = data[i:i+block_size]
        chunk_y = data[i+1:i+1+block_size]
        x[j] = torch.from_numpy(chunk_x.astype(np.int32)).long()
        y[j] = torch.from_numpy(chunk_y.astype(np.int32)).long()

    if 'cuda' in device:
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        return x.to(device), y.to(device)


def main():
    args = get_args()

    use_ddp = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    if use_ddp:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        global_rank = int(os.environ['RANK'])
        torch.cuda.set_device(local_rank)
        device = f'cuda:{local_rank}'
        seed_offset = global_rank
    else:
        local_rank = 0
        global_rank = 0
        device = args.device
        seed_offset = 0

    seed_everything(args.seed + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    if global_rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)
    
    data_dir = os.path.join('/scratch1/mengxiwu/nanoGPT/data', args.dataset)
    init_data_loader('train', data_dir, args.block_size)
    
    meta_path = os.path.join(data_dir, 'meta.pkl')
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta['vocab_size']
        if global_rank == 0:
            print(f"Found vocab_size = {meta_vocab_size} (inside {meta_path})")

    param_type = 'mup'
    mup_multiplier = args.n_embd / 512.0

    model_args = dict(
        n_layer=args.n_layer, n_head=args.n_head, n_kv_head=args.n_kv_head, n_embd=args.n_embd,
        block_size=args.block_size, bias=args.bias, vocab_size=meta_vocab_size or 50304,
        dropout=args.dropout, mup=True, mup_multiplier=mup_multiplier, init_std=args.init_std,
        impl=impl_dict['mengxi_impl'], normalization="LayerNorm", use_moe=args.use_moe,
        num_experts=args.num_experts, router_topk=args.router_topk
    )
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.to(device)

    if args.compile:
        if global_rank == 0:
            print("Compiling the model... (this may take a minute)")
        model = torch.compile(model)

    if use_ddp:
        model = DDP(model, device_ids=[local_rank])

    model_for_optimizer = model.module if use_ddp else model
    optimizer = model_for_optimizer.configure_optimizers(
        weight_decay=args.weight_decay, learning_rate=args.learning_rate,
        betas=(args.beta1, args.beta2), eps=args.eps,
        device_type=device_type, adaptive_optimizer=False
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))

    losses = []
    t0 = time.time()
    for step in range(args.max_iters):
        x, y = get_batch('train', args.batch_size, args.block_size, device)
        
        with ctx:
            outputs = model(x, y)
            loss = outputs[1] if isinstance(outputs, tuple) and outputs[1] is not None else outputs[0]
            loss = loss / args.gradient_accumulation_steps
        
        scaler.scale(loss).backward()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.grad_clip > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if (step + 1) % 10 == 0 and global_rank == 0:
            loss_val = loss.item() * args.gradient_accumulation_steps
            losses.append(loss_val)
            dt = time.time() - t0
            t0 = time.time()
            
            mfu = -1.0
            if hasattr(model_for_optimizer, 'estimate_mfu'):
                mfu = model_for_optimizer.estimate_mfu(args.gradient_accumulation_steps, dt)
            
            print(f"Step {step+1}/{args.max_iters} | Loss: {loss_val:.4f} | Time: {dt*1000:.2f}ms | MFU: {mfu*100:.2f}%")


    if global_rank == 0:
        out_dir_full = os.path.join(args.out_dir, param_type)
        os.makedirs(out_dir_full, exist_ok=True)
        
        log_file_name = f"width{args.n_embd}_lr{args.learning_rate:.5f}_wd{args.weight_decay:.5f}_seed{args.seed}.csv"
        
        df = pd.DataFrame({
            'step': np.arange(len(losses)) * 10,
            'loss': losses,
            'width': args.n_embd,
            'lr': args.learning_rate,
            'wd': args.weight_decay,
            'seed': args.seed,
            'param_type': param_type
        })
        df.to_csv(os.path.join(out_dir_full, log_file_name), index=False)
        print(f"Saved training log to {os.path.join(out_dir_full, log_file_name)}")

    if use_ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()