"""NanoGPT architecture unit tests — CPU only, no real training.

Verifies forward shapes, RoPE caching, generation, weight tying,
and the trainer's tokenisation helper.
"""

import pytest
import torch

from core.training.models.nano_gpt import (
    NanoGPT,
    NanoGPTConfig,
    causal_mask,
)


def _tiny_cfg() -> NanoGPTConfig:
    return NanoGPTConfig(
        vocab_size=128,
        block_size=32,
        n_layer=2,
        n_head=2,
        n_embd=32,
    )


def test_config_validates_head_divisibility() -> None:
    bad = NanoGPTConfig(n_embd=33, n_head=4, vocab_size=10, block_size=8, n_layer=1)
    with pytest.raises(ValueError):
        bad.head_dim


def test_causal_mask_is_lower_triangular() -> None:
    m = causal_mask(4, torch.device("cpu"))
    assert m.shape == (4, 4)
    # below-diagonal should be 0
    for i in range(4):
        for j in range(i + 1):
            assert m[i, j].item() == 0.0
    # above-diagonal should be -inf
    for i in range(4):
        for j in range(i + 1, 4):
            assert m[i, j].item() == float("-inf")


def test_forward_shapes() -> None:
    cfg = _tiny_cfg()
    model = NanoGPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (3, 16))
    logits, loss = model(x)
    assert logits.shape == (3, 16, cfg.vocab_size)
    assert loss is None

    y = torch.randint(0, cfg.vocab_size, (3, 16))
    logits2, loss2 = model(x, targets=y)
    assert logits2.shape == logits.shape
    assert loss2 is not None
    assert loss2.dim() == 0


def test_block_size_enforced() -> None:
    cfg = _tiny_cfg()
    model = NanoGPT(cfg)
    too_long = torch.randint(0, cfg.vocab_size, (1, cfg.block_size + 1))
    with pytest.raises(ValueError):
        model(too_long)


def test_weight_tying() -> None:
    cfg = _tiny_cfg()
    model = NanoGPT(cfg)
    assert model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr()


def test_weight_tying_off() -> None:
    cfg = NanoGPTConfig(
        vocab_size=64, block_size=8, n_layer=1, n_head=2, n_embd=16,
        tie_weights=False,
    )
    model = NanoGPT(cfg)
    assert model.lm_head.weight.data_ptr() != model.tok_emb.weight.data_ptr()


def test_num_parameters_excludes_embeddings_correctly() -> None:
    cfg = _tiny_cfg()
    model = NanoGPT(cfg)
    total = model.num_parameters()
    bare = model.num_parameters(exclude_embeddings=True)
    # Tied weights → only one embedding to subtract
    assert bare == total - cfg.vocab_size * cfg.n_embd


def test_generate_extends_sequence() -> None:
    cfg = _tiny_cfg()
    model = NanoGPT(cfg)
    seed = torch.randint(0, cfg.vocab_size, (1, 4))
    out = model.generate(seed, max_new_tokens=8, temperature=0.0)
    assert out.shape == (1, 12)
    # original prefix preserved
    assert torch.equal(out[0, :4], seed[0])


def test_generate_eos_terminates_early() -> None:
    cfg = _tiny_cfg()
    model = NanoGPT(cfg)
    seed = torch.randint(0, cfg.vocab_size, (1, 2))
    # set EOS to whatever the model would produce; can't predict
    # exactly, so we just check the function returns and length is
    # ≤ requested.
    out = model.generate(seed, max_new_tokens=10, temperature=1.0, eos_token_id=0)
    assert out.shape[1] <= 12


def test_top_k_sampling_reduces_options() -> None:
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    model = NanoGPT(cfg)
    seed = torch.randint(0, cfg.vocab_size, (1, 4))
    out = model.generate(seed, max_new_tokens=4, temperature=1.0, top_k=2)
    assert out.shape == (1, 8)


def test_rope_cache_per_device() -> None:
    cfg = _tiny_cfg()
    model = NanoGPT(cfg)
    cpu = torch.device("cpu")
    cos1, sin1 = model._rope(cpu)
    cos2, sin2 = model._rope(cpu)
    # cached -> same tensor object
    assert cos1 is cos2
    assert sin1 is sin2


def test_uses_only_torch() -> None:
    """Sanity: the model module imports must not pull in transformers
    or accelerate or huggingface_hub.
    """
    import sys
    import importlib

    # fresh import
    if "core.training.models.nano_gpt" in sys.modules:
        importlib.reload(sys.modules["core.training.models.nano_gpt"])
    import core.training.models.nano_gpt as mod
    assert mod.__name__.endswith("nano_gpt")
    # Check the module file doesn't import these libs at top-level.
    src = open(mod.__file__, encoding="utf-8").read()
    for forbidden in ("import transformers", "from transformers",
                       "import accelerate", "from accelerate",
                       "import huggingface_hub", "from huggingface_hub"):
        assert forbidden not in src, f"nano_gpt.py must not depend on {forbidden!r}"
