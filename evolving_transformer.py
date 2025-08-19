
# evolving_transformer.py
# A compact, configurable GPT-style Transformer with modern upgrades.
# PyTorch >= 2.1 recommended.

import math
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Utils: Norms & Activations
# -------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return self.weight * x


def swiglu(x: torch.Tensor) -> torch.Tensor:
    x, gate = x.chunk(2, dim=-1)
    return F.silu(gate) * x


# -------------------------
# Rotary Positional Embedding
# -------------------------

class RotaryEmbedding:
    def __init__(self, dim: int, base: float = 10000.0, max_seq_len: int = 4096, device=None):
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        t = torch.arange(max_seq_len, device=device).float()
        freqs = torch.einsum('i,j->ij', t, inv_freq)  # [T, dim/2]
        self.cos = torch.cos(freqs)
        self.sin = torch.sin(freqs)

    def apply(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # x: [B, H, T, D]; D even
        cos = self.cos[positions]  # [B,T,D/2]
        sin = self.sin[positions]
        cos = cos.unsqueeze(1)   # [B,1,T,D/2]
        sin = sin.unsqueeze(1)

        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return x_rotated.flatten(-2)


# -------------------------
# Config
# -------------------------

@dataclass
class ModelConfig:
    vocab_size: int = 256       # byte-level by default
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 2         # GQA/MQA style; <= n_heads
    d_ff: int = 2048
    max_seq_len: int = 1024
    dropout: float = 0.0
    attn_dropout: float = 0.0
    prenorm: bool = True
    norm_type: str = "rmsnorm"  # "layernorm" | "rmsnorm"
    act: str = "swiglu"         # "gelu" | "swiglu"
    use_rope: bool = True
    rope_base: float = 10000.0
    use_flash: bool = True      # use sDPA fastpath
    residual_scale: Optional[float] = 1/math.sqrt(2)
    tie_embeddings: bool = True
    use_moe: bool = False
    moe_experts: int = 4
    moe_top1: bool = True
    use_cache: bool = True

def make_norm(dim: int, cfg: ModelConfig) -> nn.Module:
    if cfg.norm_type == "rmsnorm":
        return RMSNorm(dim)
    return nn.LayerNorm(dim)

def make_act(cfg: ModelConfig):
    if cfg.act == "swiglu":
        return "swiglu"
    return nn.GELU()

# -------------------------
# Attention
# -------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.n_heads = cfg.n_heads
        self.n_kv = cfg.n_kv_heads
        assert cfg.d_model % cfg.n_heads == 0, "d_model must be divisible by n_heads"
        self.head_dim = d // cfg.n_heads
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"

        self.W_q = nn.Linear(d, cfg.n_heads * self.head_dim, bias=False)
        self.W_k = nn.Linear(d, self.n_kv * self.head_dim, bias=False)
        self.W_v = nn.Linear(d, self.n_kv * self.head_dim, bias=False)
        self.W_o = nn.Linear(cfg.n_heads * self.head_dim, d, bias=False)

        self.dropout = nn.Dropout(cfg.attn_dropout) if cfg.attn_dropout > 0 else nn.Identity()
        self.rope = RotaryEmbedding(self.head_dim, base=cfg.rope_base,
                                    max_seq_len=cfg.max_seq_len) if cfg.use_rope else None

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_kv == self.n_heads:
            return x
        repeat = self.n_heads // self.n_kv
        return x.repeat_interleave(repeat, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ):
        B, T, D = x.shape
        Hd = self.head_dim

        q = self.W_q(x).view(B, T, self.n_heads, Hd).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_kv,   Hd).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_kv,   Hd).transpose(1, 2)

        if self.rope is not None:
            if position_ids is None:
                position_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
            q = self.rope.apply(q, position_ids)
            k = self.rope.apply(k, position_ids)

        if past_kv is not None:
            pk, pv = past_kv  # [B,H_kv,Tpast,Hd]
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)

        k_rep = self._repeat_kv(k)
        v_rep = self._repeat_kv(v)

        if self.cfg.use_flash:
            attn = F.scaled_dot_product_attention(
                q, k_rep, v_rep, attn_mask=attn_mask,
                dropout_p=self.cfg.attn_dropout if self.training else 0.0,
                is_causal=(attn_mask is None)
            )
        else:
            scores = torch.matmul(q, k_rep.transpose(-2, -1)) / math.sqrt(Hd)
            if attn_mask is not None:
                scores = scores + attn_mask
            else:
                causal = torch.triu(torch.ones(T, k_rep.size(-2), device=x.device), diagonal=1).bool()
                scores = scores.masked_fill(causal, float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            attn = torch.matmul(attn, v_rep)

        y = attn.transpose(1, 2).contiguous().view(B, T, self.n_heads * Hd)
        y = self.W_o(y)
        present = (k, v) if use_cache else None
        return y, present

# -------------------------
# Feedforward / MoE
# -------------------------

class FeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        inner = cfg.d_ff
        if cfg.act == "swiglu":
            self.fc1 = nn.Linear(d, 2 * inner, bias=False)
            self.fc2 = nn.Linear(inner, d, bias=False)
            self.act = "swiglu"
        else:
            self.fc1 = nn.Linear(d, inner, bias=False)
            self.fc2 = nn.Linear(inner, d, bias=False)
            self.act = nn.GELU()
        self.drop = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.act == "swiglu":
            h = swiglu(self.fc1(x))
        else:
            h = self.act(self.fc1(x))
        h = self.drop(h)
        return self.fc2(h)

class SwitchFFN(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.gate = nn.Linear(cfg.d_model, cfg.moe_experts, bias=False)
        self.experts = nn.ModuleList([FeedForward(cfg) for _ in range(cfg.moe_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.gate(x)  # [B,T,E]
        if self.cfg.moe_top1:
            idx = torch.argmax(logits, dim=-1)  # [B,T]
            outs = torch.stack([e(x) for e in self.experts], dim=-1)  # [B,T,D,E]
            idx_exp = idx.unsqueeze(-1).unsqueeze(-1).expand(x.size(0), x.size(1), x.size(2), 1)
            y = outs.gather(-1, idx_exp).squeeze(-1)  # [B,T,D]
            return y
        else:
            probs = torch.softmax(logits, dim=-1)     # [B,T,E]
            outs = torch.stack([e(x) for e in self.experts], dim=-1)  # [B,T,D,E]
            y = torch.einsum('bte,btde->btd', probs, outs)
            return y

# -------------------------
# Transformer Block
# -------------------------

class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.attn = MultiHeadAttention(cfg)
        self.ffn = SwitchFFN(cfg) if cfg.use_moe else FeedForward(cfg)
        self.norm1 = make_norm(cfg.d_model, cfg)
        self.norm2 = make_norm(cfg.d_model, cfg)
        self.drop = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

    def residual(self, x: torch.Tensor, sub: torch.Tensor) -> torch.Tensor:
        if self.cfg.residual_scale is not None:
            return x + self.cfg.residual_scale * sub
        return x + sub

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ):
        if self.cfg.prenorm:
            a_in = self.norm1(x)
            a_out, present = self.attn(a_in, attn_mask, position_ids, past_kv, use_cache=use_cache)
            x = self.residual(x, self.drop(a_out))
            m_in = self.norm2(x)
            m_out = self.ffn(m_in)
            x = self.residual(x, self.drop(m_out))
        else:
            a_out, present = self.attn(x, attn_mask, position_ids, past_kv, use_cache=use_cache)
            x = self.norm1(self.residual(x, self.drop(a_out)))
            m_out = self.ffn(x)
            x = self.norm2(self.residual(x, self.drop(m_out)))
        return x, present

# -------------------------
# Full Model
# -------------------------

class EvolvingTransformerLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model) if not cfg.use_rope else None
        self.drop = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = make_norm(cfg.d_model, cfg)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

    def forward(
        self,
        input_ids: torch.Tensor,        # [B,T]
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: Optional[bool] = None
    ):
        B, T = input_ids.shape
        cfg = self.cfg
        use_cache = cfg.use_cache if use_cache is None else use_cache

        pos_ids = position_ids
        if not cfg.use_rope:
            if pos_ids is None:
                pos_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
            h = self.tok_emb(input_ids) + self.pos_emb(pos_ids)
        else:
            h = self.tok_emb(input_ids)
            if pos_ids is None:
                pos_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)

        h = self.drop(h)

        presents = []
        for i, blk in enumerate(self.blocks):
            past_kv = past_key_values[i] if (past_key_values is not None) else None
            h, present = blk(h, attn_mask=None, position_ids=pos_ids, past_kv=past_kv, use_cache=use_cache)
            if use_cache:
                presents.append(present)

        h = self.final_norm(h)
        logits = self.lm_head(h)
        return (logits, presents) if use_cache else (logits, None)


def _smoke_test():
    cfg = ModelConfig()
    model = EvolvingTransformerLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, _ = model(x)
    print("ok", logits.shape)

if __name__ == "__main__":
    _smoke_test()
