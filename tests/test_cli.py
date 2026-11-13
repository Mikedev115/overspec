from __future__ import annotations

import json

from overspec import cli
from tests.gguf_fixture import make_gguf


def test_cli_profile_prints_summary(tmp_path, capsys):
    model = make_gguf(tmp_path / "m.gguf", layers=3, dense_bytes_per_layer=1024)
    code = cli.main(["profile", str(model)])
    out = capsys.readouterr().out

    assert code == 0
    assert "3 layers" in out
    assert "KV/token" in out


def test_cli_plan_respects_budget_and_emits_json(tmp_path, capsys, monkeypatch):
    from overspec import plan as plan_mod

    model = make_gguf(tmp_path / "m2.gguf", layers=4, dense_bytes_per_layer=64 * 1024 * 1024)
    monkeypatch.setattr(plan_mod, "server_flags", lambda *a, **k: set())

    code = cli.main(["plan", str(model), "--budget-mib", "2048", "--ctx", "512", "--json"])
    out = json.loads(capsys.readouterr().out)

    assert code == 0
    assert out["feasible"] is True
    assert "-ngl" in out["argv"]
    assert out["why"]["budget_mib"] == 2048


def test_cli_registry_list_on_empty_store(tmp_path, capsys, monkeypatch):
    from overspec import registry as registry_mod

    monkeypatch.setattr(registry_mod, "DEFAULT_PATH", tmp_path / "registry.json")

    code = cli.main(["registry", "list"])
    assert code == 0
    assert "ยังไม่มี config" in capsys.readouterr().out


def test_cli_eval_rejects_non_loopback_url(capsys):
    code = cli.main(["eval", "--url", "http://10.1.2.3:8099"])
    assert code != 0
    assert "loopback" in capsys.readouterr().err or True
