
# sample_shakespeare.py
# Load a checkpoint and generate text using KV-cache, temperature, top-p, and repetition penalties.
# Usage:
#   python sample_shakespeare.py --checkpoint ./checkpoints/best_ckpt.pt --prompt "To be, or not to be" --max_new_tokens 400 --temperature 0.9 --top_p 0.9

import argparse, torch
from evolving_transformer import ModelConfig, EvolvingTransformerLM

def encode_bytes(s: str):
    return list(s.encode("utf-8", errors="ignore"))

def decode_bytes(ids):
    ids = [i for i in ids if 0 <= i < 256]
    return bytes(ids).decode("utf-8", errors="ignore")

@torch.no_grad()
def generate(model, cfg, prompt: str, max_new_tokens: int = 300, temperature: float = 1.0,
             top_p: float = 1.0, freq_penalty: float = 0.0, presence_penalty: float = 0.0,
             device="cuda" if torch.cuda.is_available() else "cpu"):
    model.eval()
    use_cache = True
    input_ids = torch.tensor([encode_bytes(prompt)], dtype=torch.long, device=device)
    B, T = input_ids.shape

    position_ids = torch.arange(T, device=device).unsqueeze(0)
    logits, past = model(input_ids, position_ids=position_ids, use_cache=use_cache)

    generated = list(input_ids[0].tolist())
    total_len = T

    for step in range(max_new_tokens):
        last_logits = logits[:, -1, :]
        if freq_penalty > 0.0 or presence_penalty > 0.0:
            counts = torch.bincount(torch.tensor(generated, device=device), minlength=cfg.vocab_size).float()
            last_logits = last_logits - freq_penalty * counts - presence_penalty * (counts > 0).float()

        if temperature != 1.0:
            last_logits = last_logits / max(1e-5, temperature)

        probs = torch.softmax(last_logits, dim=-1)

        if 0.0 < top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)
            mask = cum > top_p
            mask[:, 0] = False
            sorted_probs = torch.where(mask, torch.zeros_like(sorted_probs), sorted_probs)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            next_id = torch.multinomial(sorted_probs, num_samples=1)
            next_token = sorted_idx.gather(-1, next_id)
        else:
            next_token = torch.multinomial(probs, num_samples=1)

        token_id = int(next_token.item())
        generated.append(token_id)

        x = next_token.view(1, 1)
        total_len += 1
        position_ids = torch.tensor([[total_len - 1]], device=device)
        logits, past = model(x, position_ids=position_ids, past_key_values=past, use_cache=use_cache)

    return decode_bytes(generated)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="ROMEO: ")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--freq_penalty", type=float, default=0.0)
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ModelConfig(**ckpt["cfg"])
    cfg.use_cache = True
    model = EvolvingTransformerLM(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])

    out = generate(model, cfg, prompt=args.prompt, max_new_tokens=args.max_new_tokens,
                   temperature=args.temperature, top_p=args.top_p,
                   freq_penalty=args.freq_penalty, presence_penalty=args.presence_penalty,
                   device=device)
    print(out)

if __name__ == "__main__":
    main()
