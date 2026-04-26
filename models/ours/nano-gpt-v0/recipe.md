# nanoGPT recipe

From-scratch training run by `core.training.nano_gpt_trainer.train_nano_gpt`.

## Architecture

- vocab_size: 151665
- block_size: 384
- n_layer × n_head × n_embd = 6 × 8 × 384
- params: 68,861,184 (68.86M)

## Training

- num_steps: 1500
- batch_size: 16
- tokens_seen: 9,216,000
- lr: 0.0003 (warmup 50, cosine decay to 10%)
- device: cuda
- wall: 294.8s

## Loss

- first: 12.014434814453125
- final: 0.014977529644966125
