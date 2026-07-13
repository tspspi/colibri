from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

MAGIC = b"ALBTV1\x00\x00"
META = struct.Struct("<ii")


def read_samples(path: str | Path) -> np.ndarray:
    raw = Path(path).read_bytes()
    if len(raw) < len(MAGIC) or raw[: len(MAGIC)] != MAGIC:
        raise ValueError(f"{path} is not a colibrialbate sample file")
    off = len(MAGIC)
    rows = []
    dim = None
    while off < len(raw):
        if off + META.size > len(raw):
            raise ValueError(f"{path} has a truncated sample header")
        layer, rec_dim = META.unpack_from(raw, off)
        off += META.size
        byte_len = rec_dim * 4
        if rec_dim < 1 or off + byte_len > len(raw):
            raise ValueError(f"{path} has a truncated sample payload")
        vec = np.frombuffer(raw, dtype="<f4", count=rec_dim, offset=off).astype(np.float32, copy=True)
        off += byte_len
        if dim is None:
            dim = rec_dim
        elif rec_dim != dim:
            raise ValueError(f"{path} mixes hidden sizes ({dim} and {rec_dim})")
        rows.append((layer, vec))
    if not rows:
        raise ValueError(f"{path} contains no samples")
    layers = {layer for layer, _ in rows}
    if len(layers) != 1:
        raise ValueError(f"{path} mixes layers: {sorted(layers)}")
    return np.stack([vec for _, vec in rows], axis=0)


def write_direction(path: str | Path, direction: np.ndarray) -> None:
    direction.astype("<f4", copy=False).tofile(Path(path))


def write_manifest(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")
