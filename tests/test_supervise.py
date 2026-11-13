from __future__ import annotations

import sys
from pathlib import Path

from overspec import supervise as sv
from overspec.plan import Plan

FAKE_SERVER = Path(__file__).resolve().parent / "fake_server.py"


def test_parse_allocated_mib():
    log = """0.04.844.465 I load_tensors: offloading 31 repeating layers to GPU
0.04.844.470 I load_tensors:        CUDA0 model buffer size =  4458.03 MiB
0.04.844.471 I load_tensors:    CUDA_Host model buffer size =   804.97 MiB
"""
    assert sv.parse_allocated_mib(log) == 4458
    assert sv.parse_allocated_mib("no buffer here") is None


def test_spill_ratio_matches_measured_case():
    # เคสจริง: จอง 4458 MiB แต่ VRAM เพิ่มจริง 266 MiB
    ratio = sv.compute_spill_ratio(4458, 266)
    assert ratio is not None
    assert 0.93 < ratio < 0.95
    assert sv.compute_spill_ratio(4458, 4400) < sv.SPILL_THRESHOLD
    assert sv.compute_spill_ratio(None, 100) is None
    assert sv.compute_spill_ratio(4458, 0) == 1.0


def test_ladder_walks_ctx_ngl_cache():
    plan = Plan(argv=["-c", "4096", "-fa", "on", "-t", "12", "-ngl", "32",
                      "--moe-cache-mib", "2048"])

    step0 = sv.ladder(plan, 0)
    assert step0.argv[step0.argv.index("-c") + 1] == "2048"

    step1 = sv.ladder(plan, 1)
    assert step1.argv[step1.argv.index("-ngl") + 1] == "28"

    step2 = sv.ladder(plan, 2)
    assert step2.argv[step2.argv.index("--moe-cache-mib") + 1] == "1024"

    assert sv.ladder(plan, 3) is None


def test_ladder_stops_when_nothing_left_to_shrink():
    assert sv.ladder(Plan(argv=["-c", "2048"]), 0) is None
    assert sv.ladder(Plan(argv=["-ngl", "2"]), 1) is None
    assert sv.ladder(Plan(argv=["--moe-cache-mib", "600"]), 2) is None


class FakeSnap:
    def __init__(self, resident_mib: int):
        self.resident_mib = resident_mib

    def our_process_mib(self, pid: int) -> int:
        return self.resident_mib

    def budget_source(self) -> str:
        return "fake"


def test_run_once_detects_resident_and_tok_s(tmp_path):
    port = 8123
    plan = Plan(argv=["-c", "512"])
    cmd = [sys.executable, str(FAKE_SERVER), "--port", str(port), "--allocated", "4458.03"]

    result = sv.run_once(plan, "model.gguf", port=port, startup_timeout=30,
                         workdir=tmp_path, server_cmd=cmd,
                         snapshot_fn=lambda: FakeSnap(4400))

    assert result.verdict == "ok"
    assert result.allocated_mib == 4458
    assert result.resident_mib == 4400
    assert result.spill_ratio is not None and result.spill_ratio < sv.SPILL_THRESHOLD
    assert result.tok_per_sec == 12.5
    assert Path(result.log_path).exists()


def test_run_once_flags_spill(tmp_path):
    port = 8124
    plan = Plan(argv=["-c", "512"])
    cmd = [sys.executable, str(FAKE_SERVER), "--port", str(port), "--allocated", "4458.03"]

    result = sv.run_once(plan, "model.gguf", port=port, startup_timeout=30,
                         workdir=tmp_path, server_cmd=cmd,
                         snapshot_fn=lambda: FakeSnap(1000))

    assert result.verdict == "spill"
    assert result.spill_ratio > sv.SPILL_THRESHOLD


def test_build_cmd_uses_server_cmd_when_given():
    cmd = sv.build_cmd(Path("llama-server.exe"), "m.gguf", 8080, ["-c", "4096"],
                       server_cmd=["fake", "--port", "8123"])
    assert cmd == ["fake", "--port", "8123"]

    default = sv.build_cmd(Path("llama-server.exe"), "m.gguf", 8080, ["-c", "4096"])
    assert default[:3] == ["llama-server.exe", "-m", "m.gguf"]
    assert "-c" in default
