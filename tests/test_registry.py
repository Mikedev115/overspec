from __future__ import annotations

from overspec.registry import Record, Registry, config_key


def _rec(key: str, model: str = "qwen9b.gguf", status: str = "candidate") -> Record:
    return Record(key=key, model=model, argv=["-c", "4096"], status=status,
                  tok_per_sec=20.0, spill_ratio=0.02)


def test_config_key_is_stable_and_argv_sensitive():
    a = config_key("qwen9b.gguf", ["-c", "4096", "-ngl", "32"])
    b = config_key("qwen9b.gguf", ["-c", "4096", "-ngl", "32"])
    c = config_key("qwen9b.gguf", ["-c", "2048", "-ngl", "32"])

    assert a == b
    assert a != c
    assert len(a) == 12


def test_config_key_ignores_directory(tmp_path):
    a = config_key(str(tmp_path / "a" / "qwen9b.gguf"), ["-c", "4096"])
    b = config_key(str(tmp_path / "b" / "qwen9b.gguf"), ["-c", "4096"])
    assert a == b


def test_add_persists_and_reloads(tmp_path):
    path = tmp_path / "registry.json"
    Registry(path).add(_rec("abc123"))

    reg = Registry(path)
    rec = reg.get("abc123")
    assert rec is not None
    assert rec.tok_per_sec == 20.0
    assert rec.model == "qwen9b.gguf"


def test_promote_keeps_one_production_per_model(tmp_path):
    reg = Registry(tmp_path / "registry.json")
    reg.add(_rec("aaa111"))
    reg.add(_rec("bbb222"))

    reg.promote("aaa111")
    reg.promote("bbb222")

    prod = reg.production("qwen9b.gguf")
    assert [p.key for p in prod] == ["bbb222"]
    assert reg.get("aaa111").status == "candidate"


def test_production_kept_when_record_is_updated(tmp_path):
    path = tmp_path / "registry.json"
    reg = Registry(path)
    reg.add(_rec("ccc333"))
    reg.promote("ccc333")

    reg.add(_rec("ccc333"))  # วัฒน์ซ้ำ config เดิม
    assert Registry(path).get("ccc333").status == "production"


def test_reject_marks_status(tmp_path):
    reg = Registry(tmp_path / "registry.json")
    reg.add(_rec("ddd444"))
    assert reg.reject("ddd444").status == "rejected"
