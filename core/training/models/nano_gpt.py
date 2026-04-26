"""nanoGPT — decoder-only Transformer in pure PyTorch.

PROVENANCE
  reference: Andrej Karpathy "nanoGPT" — https://github.com/karpathy/nanoGPT
  reference: GPT-2 modeling code, paper "Language Models are Unsupervised
             Multitask Learners" (Radford et al., 2019)
  our changes: simplified to ~250 lines, RMSNorm + SwiGLU + RoPE swap
               from the GPT-2 LayerNorm + GELU + learned-pos defaults to
               match the Qwen2 family our LoRA work targets, so we can
               compare like-for-like architectures
  license: Apache 2.0 (compatible with both refs)
  no transformers / accelerate / huggingface_hub dependencies — torch only
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NanoGPTConfig:
    vocab_size: int = 32_000
    block_size: int = 512        # context length
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 384
    dropout: float = 0.0
    rope_theta: float = 10_000.0
    tie_weights: bool = True

    @property
    def head_dim(self) -> int:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        return self.n_embd // self.n_head

    def to_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "block_size": self.block_size,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_embd": self.n_embd,
            "dropout": self.dropout,
            "rope_theta": self.rope_theta,
            "tie_weights": self.tie_weights,
        }


def causal_mask(T: int, device: torch.device) -> torch.Tensor:
    """Lower-triangular mask: row i can only attend to columns 0..i.
    Returned as additive float (-inf above diagonal, 0 below) so it
    can be added directly to attention logits.
    """
    m = torch.full((T, T), float("-inf"), device=device)
    return torch.triu(m, diagonal=1)


def _build_rope(head_dim: int, max_len: int, theta: float, device: torch.device):
    """Pre-compute RoPE sin/cos tables. Same construction as Llama/Qwen2.
    """
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(max_len, device=device).float()
    freqs = torch.outer(pos, inv)              # (max_len, head_dim/2)
    return torch.cos(freqs), torch.sin(freqs)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotary position embedding. x is (B, H, T, head_dim)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[: x.size(-2)].unsqueeze(0).unsqueeze(0)   # (1,1,T,half)
    sin = sin[: x.size(-2)].unsqueeze(0).unsqueeze(0)
    rot = torch.empty_like(x)
    rot[..., 0::2] = x1 * cos - x2 * sin
    rot[..., 1::2] = x1 * sin + x2 * cos
    return rot


class RMSNorm(nn.Module):
    """Root-mean-square LayerNorm (LLaMA / Qwen2 style)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True).add_(self.eps).rsqrt_()
        return (x * norm) * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: NanoGPTConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = cfg.dropout

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        # PyTorch >=2.0 fused attention with causal mask
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class SwiGLU(nn.Module):
    """Gated MLP with SiLU activation (LLaMA / Qwen2 standard)."""

    def __init__(self, n_embd: int, hidden_mult: int = 4) -> None:
        super().__init__()
        # Round to multiple of 64 to keep tensor shapes nice.
        hidden = int(((n_embd * hidden_mult * 2 / 3 + 63) // 64) * 64)
        self.gate = nn.Linear(n_embd, hidden, bias=False)
        self.up = nn.Linear(n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: NanoGPTConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg.n_embd)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class NanoGPT(nn.Module):
    """Decoder-only transformer. No external library — torch only.

    Forward:  (B, T) token ids → (B, T, vocab_size) logits
    Generate: greedy or temperature sampling, single-token-at-a-time
              with KV-less re-encode (simple; we trade speed for
              source clarity).
    """

    def __init__(self, cfg: NanoGPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        self._rope_cache: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _rope(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if device not in self._rope_cache:
            self._rope_cache[device] = _build_rope(
                self.cfg.head_dim, self.cfg.block_size,
                self.cfg.rope_theta, device,
            )
        return self._rope_cache[device]

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(
                f"sequence length {T} exceeds block_size {self.cfg.block_size}"
            )

        x = self.tok_emb(idx)
        cos, sin = self._rope(idx.device)
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            cropped = idx[:, -self.cfg.block_size :]
            logits, _ = self(cropped)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            if temperature == 0.0:
                next_id = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            if eos_token_id is not None and (next_id == eos_token_id).all():
                break
        return idx

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        params = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            params -= self.tok_emb.weight.numel()
            if not self.cfg.tie_weights:
                params -= self.lm_head.weight.numel()
        return params


@dataclass
class GenerateConfig:
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_k: int | None = 50
    eos_token_id: int | None = None
    seeds: list[int] = field(default_factory=list)
