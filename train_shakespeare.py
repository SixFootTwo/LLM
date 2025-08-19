
# train_shakespeare.py (revised logging)
# Usage:
#   python train_shakespeare.py --steps 200 --block_size 256 --batch_size 8 --accum 1 --log_every 10
#
# Adds:
# - --log_every
# - startup info prints
# - fused AdamW fallback

import argparse, os, math, time, random, urllib.request
from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from evolving_transformer import ModelConfig, EvolvingTransformerLM

URL = "https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt"

def download_dataset(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print(f"Downloading Shakespeare to {path} ...")
        urllib.request.urlretrieve(URL, path)
        print("Done.")
    else:
        print(f"Dataset already exists at {path}")

class ByteDataset(Dataset):
    def __init__(self, data_bytes: bytes, block_size: int):
        super().__init__()
        self.data = torch.tensor(list(data_bytes), dtype=torch.long)  # values 0..255
        self.block = block_size

    def __len__(self):
        return max(1, self.data.size(0) - self.block - 1)

    def __getitem__(self, idx):
        start = random.randint(0, self.data.size(0) - self.block - 2)
        x = self.data[start:start+self.block]
        y = self.data[start+1:start+self.block+1]
        return x, y

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def num_params(model):
    return sum(p.numel() for p in model.parameters())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/shakespeare.txt")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--accum", type=int, default=2, help="gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_kv_heads", type=int, default=2)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    # 1) Data
    download_dataset(args.data_path)
    with open(args.data_path, "rb") as f:
        data_bytes = f.read()
    N = len(data_bytes)
    split = int(0.9 * N)
    train_bytes = data_bytes[:split]
    val_bytes = data_bytes[split:]

    print(f"[info] Dataset bytes: total={N:,} | train={len(train_bytes):,} | val={len(val_bytes):,}")

    train_ds = ByteDataset(train_bytes, args.block_size)
    val_ds   = ByteDataset(val_bytes,   args.block_size)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, drop_last=True)

    # 2) Model config
    cfg = ModelConfig(
        vocab_size=256,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        attn_dropout=0.0 if args.dropout == 0.0 else min(0.1, args.dropout),
        prenorm=True,
        norm_type="rmsnorm",
        act="swiglu",
        use_rope=True,
        use_flash=True,
        residual_scale=1/math.sqrt(2),
        tie_embeddings=True,
        use_moe=False,
        use_cache=False,
    )

    model = EvolvingTransformerLM(cfg).to(device)
    print(f"[info] Device: {device}")
    print(f"[info] Model params: {num_params(model):,}")
    print(f"[info] Batch: {args.batch_size} x {args.block_size} (accum={args.accum}) => tokens/step={args.batch_size*args.block_size*args.accum:,}")

    # 3) Optimizer & schedule
    try:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    except TypeError:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    autocast_dtype = torch.bfloat16 if use_bf16 else None

    def cosine_warmup(step, warmup, total):
        if step < warmup: return step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    # 4) Train loop
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val = float("inf")
    global_step = 0
    model.train()
    t0 = time.time()

    train_iter = iter(train_dl)

    print("[info] Starting training...")
    while global_step < args.steps:
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            try:
                xb, yb = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dl)
                xb, yb = next(train_iter)
            xb = xb.to(device)
            yb = yb.to(device)

            with torch.autocast(device_type='cuda', dtype=autocast_dtype, enabled=autocast_dtype is not None):
                logits, _ = model(xb)
                loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), yb.view(-1))

            loss = loss / args.accum
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = args.lr * cosine_warmup(global_step, args.warmup, args.steps)
        for g in opt.param_groups: g["lr"] = lr
        opt.step()
        global_step += 1

        if global_step % args.log_every == 0:
            tok_s = args.batch_size * args.block_size * args.accum * args.log_every / (time.time() - t0 + 1e-9)
            print(f"step {global_step:6d} | loss {loss.item()*args.accum:.4f} | lr {lr:.2e} | tok/s {tok_s:,.0f}")
            t0 = time.time()

        if global_step % args.eval_interval == 0:
            model.eval()
            with torch.no_grad():
                losses = []
                for xb, yb in val_dl:
                    xb = xb.to(device); yb = yb.to(device)
                    logits, _ = model(xb)
                    vloss = F.cross_entropy(logits.view(-1, cfg.vocab_size), yb.view(-1))
                    losses.append(vloss.item())
                val_loss = sum(losses) / len(losses)
            print(f"[eval] step {global_step} | val_loss {val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                ckpt = {
                    "model_state": model.state_dict(),
                    "cfg": asdict(cfg),
                    "val_loss": val_loss,
                    "step": global_step,
                }
                path = os.path.join(args.checkpoint_dir, "best_ckpt.pt")
                torch.save(ckpt, path)
                print(f"Saved best checkpoint to {path}")
            model.train()

        if global_step % args.save_interval == 0:
            ckpt = {
                "model_state": model.state_dict(),
                "cfg": asdict(cfg),
                "val_loss": best_val,
                "step": global_step,
            }
            path = os.path.join(args.checkpoint_dir, f"ckpt_step{global_step}.pt")
            torch.save(ckpt, path)
            print(f"Saved checkpoint to {path}")

    ckpt = {
        "model_state": model.state_dict(),
        "cfg": asdict(cfg),
        "val_loss": best_val,
        "step": global_step,
    }
    final_path = os.path.join(args.checkpoint_dir, "last_ckpt.pt")
    torch.save(ckpt, final_path)
    print(f"Training complete. Final checkpoint: {final_path}")

if __name__ == "__main__":
    main()
