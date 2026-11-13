from __future__ import annotations

from pathlib import Path

import pytest

from overspec import plan as plan_mod
from overspec.profile import read_profile
from tests.gguf_fixture import make_gguf

NO_SERVER = Path("definitely-not-a-server.exe")


def test_dense_budget_caps_layers(tmp_path):
    path = make_gguf(tmp_path / "dense.gguf", layers=8, dense_bytes_per_layer=64 * 1024 * 1024)
    prof = read_profile(str(path))

    plan = plan_mod.make_plan(prof, budget_mib=400, ctx=512, safety_mib=0,
                              server_bin=NO_SERVER, use_fit=False)
    ngl = int(plan.argv[plan.argv.index("-ngl") + 1])

    assert 0 < ngl < prof.layers
    assert plan.why["source"] == "arithmetic fallback"
    assert any("วางได้" in w for w in plan.warnings)


def test_dense_budget_large_places_all_layers(tmp_path):
    path = make_gguf(tmp_path / "small.gguf", layers=4, dense_bytes_per_layer=1024 * 1024)
    prof = read_profile(str(path))

    plan = plan_mod.make_plan(prof, budget_mib=4096, ctx=512, safety_mib=0,
                              server_bin=NO_SERVER, use_fit=False)
    ngl = int(plan.argv[plan.argv.index("-ngl") + 1])
    assert ngl == prof.layers


def test_moe_without_cache_flag_uses_ncmoe(tmp_path, monkeypatch):
    path = make_gguf(tmp_path / "moe.gguf", layers=6, dense_bytes_per_layer=1024 * 1024,
                     expert_bytes_per_layer=32 * 1024 * 1024)
    prof = read_profile(str(path))
    monkeypatch.setattr(plan_mod, "server_flags", lambda *a, **k: set())

    plan = plan_mod.make_plan(prof, budget_mib=400, ctx=512, safety_mib=0,
                              server_bin=NO_SERVER, use_fit=False)

    assert "-ncmoe" in plan.argv
    assert "--moe-cache-mib" not in plan.argv
    assert plan.why["source"].startswith("arithmetic")
    assert any("moe-cache-mib" in w for w in plan.warnings)


def test_moe_with_cache_flag_uses_cmoe(tmp_path, monkeypatch):
    path = make_gguf(tmp_path / "moe2.gguf", layers=6, dense_bytes_per_layer=1024 * 1024,
                     expert_bytes_per_layer=32 * 1024 * 1024)
    prof = read_profile(str(path))
    monkeypatch.setattr(plan_mod, "server_flags", lambda *a, **k: {"--moe-cache-mib"})

    plan = plan_mod.make_plan(prof, budget_mib=400, ctx=512, safety_mib=0,
                              server_bin=NO_SERVER, use_fit=False)

    assert "-cmoe" in plan.argv
    cache = int(plan.argv[plan.argv.index("--moe-cache-mib") + 1])
    assert 0 < cache <= int(prof.total_expert_bytes * 0.10 / (1024 * 1024)) + 1


def test_moe_that_fits_whole_stays_on_gpu(tmp_path, monkeypatch):
    path = make_gguf(tmp_path / "moe3.gguf", layers=2, dense_bytes_per_layer=1024 * 1024,
                     expert_bytes_per_layer=1024 * 1024)
    prof = read_profile(str(path))
    monkeypatch.setattr(plan_mod, "server_flags", lambda *a, **k: set())

    plan = plan_mod.make_plan(prof, budget_mib=8192, ctx=512, safety_mib=0,
                              server_bin=NO_SERVER, use_fit=False)
    assert "-ncmoe" not in plan.argv
    assert "-cmoe" not in plan.argv


def test_tiny_budget_is_infeasible(tmp_path):
    path = make_gguf(tmp_path / "dense2.gguf", layers=4, dense_bytes_per_layer=1024 * 1024)
    prof = read_profile(str(path))
    plan = plan_mod.make_plan(prof, budget_mib=1, ctx=1, safety_mib=0,
                              server_bin=NO_SERVER, use_fit=False)
    assert plan.feasible is False


def test_plan_for_model_reports_budget_source(tmp_path):
    path = make_gguf(tmp_path / "d3.gguf", layers=2, dense_bytes_per_layer=1024)
    prof = read_profile(str(path))

    class FakeSnap:
        def available_mib(self):
            return 2048

        def budget_source(self):
            return "smi"

    plan = plan_mod.plan_for_model(str(path), ctx=512, snapshot_obj=FakeSnap(),
                                   server_bin=NO_SERVER, use_fit=False)
    assert plan.why["budget_source"] == "smi"
    assert prof.layers == 2


@pytest.mark.parametrize("ctx", [512, 4096])
def test_ctx_reaches_argv(tmp_path, ctx):
    path = make_gguf(tmp_path / "d4.gguf", layers=2, dense_bytes_per_layer=1024)
    prof = read_profile(str(path))
    plan = plan_mod.make_plan(prof, budget_mib=2048, ctx=ctx, safety_mib=0,
                              server_bin=NO_SERVER, use_fit=False)
    assert plan.argv[plan.argv.index("-c") + 1] == str(ctx)
