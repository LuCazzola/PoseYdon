"""T5 embeddings of joint names, for the `joint_names` conditioner's cache.

Ported from the reference's `T5Conditioner` (`model/modules/conditioners.py`),
so a cache built here matches the one its checkpoints were trained with: the
same name clean-up, rule for rule, quirks included, and the same pooling, the
mean of T5's last hidden state over the name's tokens. Only the cache build
needs `transformers` (the `text` extra); training reads the cache.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np

T5_MODEL = "t5-base"

# The reference's tables, verbatim.
_REMOVE_PREFIXES = ("BN_Bip01", "Bip01", "BN", "NPC", "jt", "Sabrecat", "Elk")
_JAPANESE_WORDS = {
    "momo": "Thigh", "sippo": "Tail", "mune": "Chest", "hiza": "Knee", "hara": "Stomach",
    "ashi": "Leg", "hiji": "Elbow", "koshi": "Hips", "te": "Hand", "kubi": "Neck",
    "atama": "Head", "ago": "Jaw", "kata": "Shoulder",
}


def joint_name_text(name: str) -> str:
    """What T5 reads for a raw joint name: `Bip01_R_Thigh` -> `Right Thigh`.

    The reference's rules, quirks kept so its checkpoints see the same text:
    every matching prefix is stripped in turn (not only the first), a
    `Sabrecat` name also loses its last six characters, and any other
    one-letter part is dropped.
    """
    if name.startswith("Sabrecat"):
        name = name[:-6]
    for prefix in _REMOVE_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix):]

    words = []
    for part in re.split("(?=[A-Z]|_)", name):
        part = re.sub(r"[\d_]+", "", part)
        if part in ("L", "l"):
            words.append("Left")
        elif part in ("R", "r"):
            words.append("Right")
        elif len(part) <= 1:
            continue
        else:
            words.append(_JAPANESE_WORDS.get(part, "Tail" if part == "Tai" else part))
    return " ".join(words)


def encode_joint_names(names: Sequence[str], model: str = T5_MODEL) -> np.ndarray:
    """``(N, D)`` float32 T5 embeddings, one per name, on the CPU.

    A name that cleans up to nothing embeds as zeros, as in the reference.
    """
    import torch
    from transformers import T5EncoderModel, T5Tokenizer

    texts = [joint_name_text(name) for name in names]
    tokenizer = T5Tokenizer.from_pretrained(model)
    encoder = T5EncoderModel.from_pretrained(model).eval()
    inputs = tokenizer(texts, return_tensors="pt", padding=True)
    mask = inputs["attention_mask"]
    mask[[i for i, text in enumerate(texts) if not text]] = 0
    with torch.no_grad():
        hidden = encoder(**inputs).last_hidden_state
    counts = mask.sum(dim=-1).clamp(min=1).unsqueeze(-1)
    return ((hidden * mask.unsqueeze(-1)).sum(dim=-2) / counts).numpy().astype("float32")
