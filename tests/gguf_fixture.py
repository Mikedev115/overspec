"""สร้างไฟล์ GGUF ขนาดจิ๋วสำหรับเทสต์ — โครงเดียวกับของจริง (header + tensor index + data)"""
from __future__ import annotations

import struct

ALIGNMENT = 32

TYPE_U32 = 4
TYPE_STRING = 8
TYPE_ARRAY = 9

GGML_TYPE_F16 = 1
GGML_TYPE_Q4_K = 12


def _str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _kv_str(key: str, value: str) -> bytes:
    return _str(key) + struct.pack("<I", TYPE_STRING) + _str(value)


def _kv_u32(key: str, value: int) -> bytes:
    return _str(key) + struct.pack("<I", TYPE_U32) + struct.pack("<I", value)


def _kv_u32_array(key: str, values: list[int]) -> bytes:
    out = _str(key) + struct.pack("<I", TYPE_ARRAY) + struct.pack("<I", TYPE_U32)
    out += struct.pack("<Q", len(values))
    out += b"".join(struct.pack("<I", v) for v in values)
    return out


def make_gguf(path, *, layers: int = 4, dense_bytes_per_layer: int = 4096,
              expert_bytes_per_layer: int = 0, head_count: int = 8,
              head_count_kv: int = 2, embedding: int = 256,
              params_per_layer: int = 1024) -> str:
    """เขียน GGUF แล้วคืน path — dense/expert แยกตามชื่อ tensor แบบเดียวกับของจริง"""
    tensors: list[tuple[str, list[int], int, int]] = []  # name, dims, type, nbytes

    def add_layer(layer: int) -> None:
        base = f"blk.{layer}."
        if dense_bytes_per_layer:
            tensors.append((base + "attn_q.weight", [params_per_layer], GGML_TYPE_F16,
                            dense_bytes_per_layer))
        if expert_bytes_per_layer:
            for part in ("gate", "up", "down"):
                tensors.append((base + f"ffn_{part}_exps.weight", [params_per_layer],
                                GGML_TYPE_Q4_K, expert_bytes_per_layer // 3))

    for layer in range(layers):
        add_layer(layer)
    tensors.append(("output.weight", [params_per_layer], GGML_TYPE_F16, 4096))

    kv = b""
    kv += _kv_str("general.architecture", "testarch")
    kv += _kv_u32("general.alignment", ALIGNMENT)
    kv += _kv_u32("testarch.block_count", layers)
    kv += _kv_u32("testarch.attention.head_count", head_count)
    kv += _kv_u32("testarch.attention.head_count_kv", head_count_kv)
    kv += _kv_u32("testarch.embedding_length", embedding)
    kv += _kv_u32("testarch.context_length", 4096)
    n_kv = 7

    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", len(tensors)) + struct.pack("<Q", n_kv)
    header += kv

    offset = 0
    index = b""
    for name, dims, ttype, nbytes in tensors:
        index += _str(name) + struct.pack("<I", len(dims))
        index += b"".join(struct.pack("<Q", d) for d in dims)
        index += struct.pack("<I", ttype) + struct.pack("<Q", offset)
        offset += _align(nbytes)

    prefix = header + index
    pad = _align(len(prefix)) - len(prefix)
    data = b"\x00" * offset

    with open(path, "wb") as fh:
        fh.write(prefix)
        fh.write(b"\x00" * pad)
        fh.write(data)
    return str(path)


def _align(n: int, a: int = ALIGNMENT) -> int:
    return (n + a - 1) // a * a
