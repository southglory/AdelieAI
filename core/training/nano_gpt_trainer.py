"""Pure-PyTorch training loop for NanoGPT.

No transformers / accelerate / TRL. Just torch + an existing
tokenizer (we reuse Qwen2's so vocab is sane on Korean) and the
NanoGPTConfig.
"""

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from core.logging import get_logger
from core.training.models.nano_gpt import NanoGPT, NanoGPTConfig

log = get_logger("differentia.training.nano_gpt")


def encode_pairs_with_template(
    pairs: list[tuple[str, str, str]],
    tokenizer,
) -> list[list[int]]:
    """Each pair is (system, user, assistant) — render through the
    tokenizer's chat template so the EOS / role markers match what
    inference will see.
    """
    out: list[list[int]] = []
    for system, user, assistant in pairs:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
        # Render template to plain text first, then encode. This handles
        # transformers versions where tokenize=True can return raw text.
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        ids = tokenizer.encode(text, add_special_tokens=False)
        out.append(list(ids))
    return out


class TokenSequenceDataset(Dataset):
    """Chunks a flat token stream into fixed-length blocks. One pair
    is concatenated into the stream with EOS in between, so the
    block boundary doesn't fall in the middle of a turn most of
    the time.
    """

    def __init__(
        self,
        sequences: list[list[int]],
        block_size: int,
        eos_id: int,
    ) -> None:
        flat: list[int] = []
        for seq in sequences:
            flat.extend(seq)
            flat.append(eos_id)
        self.tokens = torch.tensor(flat, dtype=torch.long)
        self.block_size = block_size
        if len(self.tokens) <= block_size:
            raise ValueError(
                f"corpus too small: {len(self.tokens)} tokens, need > block_size={block_size}"
            )

    def __len__(self) -> int:
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.tokens[idx : idx + self.block_size]
        y = self.tokens[idx + 1 : idx + 1 + self.block_size]
        return x, y


def train_nano_gpt(
    *,
    sequences: list[list[int]],
    config: NanoGPTConfig,
    output_dir: str | Path,
    eos_token_id: int,
    num_steps: int = 1000,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    warmup_steps: int = 50,
    grad_clip: float = 1.0,
    log_every: int = 25,
    device: str | None = None,
) -> dict:
    """Train a freshly-initialised NanoGPT on the given token streams.
    Returns a stats dict suitable for emitting into MANIFEST.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info(
        "nano_gpt_train_start",
        extra={
            "output_dir": str(output_dir),
            "device": device,
            "vocab_size": config.vocab_size,
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "n_embd": config.n_embd,
            "block_size": config.block_size,
        },
    )

    model = NanoGPT(config).to(device)
    n_params = model.num_parameters()

    dataset = TokenSequenceDataset(sequences, config.block_size, eos_token_id)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
    )

    def _lr_at(step: int) -> float:
        if step < warmup_steps:
            return learning_rate * (step + 1) / warmup_steps
        # cosine decay to 10% of base
        progress = (step - warmup_steps) / max(num_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        cos_decay = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265)))
        return learning_rate * (0.1 + 0.9 * cos_decay.item())

    losses: list[float] = []
    t0 = time.perf_counter()
    step = 0
    iterator = iter(loader)
    while step < num_steps:
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y = next(iterator)

        x, y = x.to(device), y.to(device)
        for g in optimizer.param_groups:
            g["lr"] = _lr_at(step)

        _, loss = model(x, targets=y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        loss_val = float(loss.item())
        losses.append(loss_val)
        if (step + 1) % log_every == 0 or step == 0:
            log.info(
                "nano_gpt_step",
                extra={
                    "step": step + 1,
                    "loss": round(loss_val, 4),
                    "lr": round(optimizer.param_groups[0]["lr"], 6),
                },
            )
            print(
                f"step {step+1:>5d}/{num_steps}  "
                f"loss={loss_val:.4f}  lr={optimizer.param_groups[0]['lr']:.6f}",
                flush=True,
            )
        step += 1

    elapsed = time.perf_counter() - t0
    final_loss = losses[-1] if losses else float("nan")

    state_path = output_dir / "model.pt"
    config_path = output_dir / "config.json"
    torch.save(model.state_dict(), state_path)
    config_path.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")

    summary = {
        "output_dir": str(output_dir),
        "n_params": n_params,
        "n_params_million": round(n_params / 1e6, 2),
        "vocab_size": config.vocab_size,
        "block_size": config.block_size,
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "n_embd": config.n_embd,
        "num_steps": num_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "elapsed_seconds": round(elapsed, 1),
        "first_loss": losses[0] if losses else None,
        "final_loss": final_loss,
        "tokens_seen": num_steps * batch_size * config.block_size,
        "device": device,
    }

    manifest = {
        "model_id": output_dir.name,
        "kind": "nano-gpt-from-scratch",
        "license": "apache-2.0",
        "produced_by": "differentia-llm core.training.nano_gpt_trainer.train_nano_gpt",
        "update_policy": "diverged",
        "note": (
            "Pure-PyTorch decoder-only transformer trained from random init. "
            "No transformers/accelerate dependency. Architecture: RMSNorm + "
            "RoPE + SwiGLU (Qwen2-compatible). For learning, not production."
        ),
        **summary,
    }
    (output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "recipe.md").write_text(
        _render_recipe(summary), encoding="utf-8"
    )
    log.info("nano_gpt_train_done", extra=summary)
    return summary


def _render_recipe(s: dict) -> str:
    return (
        "# nanoGPT recipe\n\n"
        f"From-scratch training run by `core.training.nano_gpt_trainer.train_nano_gpt`.\n\n"
        "## Architecture\n\n"
        f"- vocab_size: {s['vocab_size']}\n"
        f"- block_size: {s['block_size']}\n"
        f"- n_layer × n_head × n_embd = {s['n_layer']} × {s['n_head']} × {s['n_embd']}\n"
        f"- params: {s['n_params']:,} ({s['n_params_million']}M)\n\n"
        "## Training\n\n"
        f"- num_steps: {s['num_steps']}\n"
        f"- batch_size: {s['batch_size']}\n"
        f"- tokens_seen: {s['tokens_seen']:,}\n"
        f"- lr: {s['learning_rate']} (warmup {s['warmup_steps']}, cosine decay to 10%)\n"
        f"- device: {s['device']}\n"
        f"- wall: {s['elapsed_seconds']}s\n\n"
        "## Loss\n\n"
        f"- first: {s['first_loss']}\n"
        f"- final: {s['final_loss']}\n"
    )
