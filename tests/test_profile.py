from __future__ import annotations

import pytest

from overspec.profile import read_profile
from tests.gguf_fixture import make_gguf


def test_dense_profile_layers_and_bytes(tmp_path):
    path = make_gguf(tmp_path / "dense.gguf", layers=4, dense_bytes_per_layer=4096)
    prof = read_profile(str(path))

    assert prof.layers == 4
    assert prof.is_moe is False
    assert prof.total_dense_bytes == 4 * 4096
    assert prof.total_expert_bytes == 0
    assert len(prof.bytes_per_layer()) == 4
    assert all(b > 0 for b in prof.dense_bytes)


def test_kv_bytes_per_token_from_metadata(tmp_path):
    # 4 layers · 2 kv heads · head_dim 32 · (K+V) · f16
    path = make_gguf(tmp_path / "kv.gguf", layers=4, head_count=8, head_count_kv=2,
                     embedding=256)
    prof = read_profile(str(path))
    assert prof.kv_bytes_per_token == 4 * 2 * (32 + 32) * 2


def test_moe_profile_splits_expert_bytes(tmp_path):
    path = make_gguf(tmp_path / "moe.gguf", layers=3, dense_bytes_per_layer=2048,
                     expert_bytes_per_layer=3072)
    prof = read_profile(str(path))

    assert prof.is_moe is True
    assert prof.total_expert_bytes == 3 * 3072
    assert prof.total_dense_bytes == 3 * 2048
    assert prof.expert_bytes[0] == 3072
    assert prof.bytes_per_layer()[0] == 2048 + 3072


def test_rejects_non_gguf(tmp_path):
    bogus = tmp_path / "bogus.gguf"
    bogus.write_bytes(b"NOPE" + b"\x00" * 64)
    with pytest.raises(ValueError, match="ไม่ใช่ไฟล์ GGUF"):
        read_profile(str(bogus))
