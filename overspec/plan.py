"""overspec.plan — เลือก argument ของ llama-server จาก profile + งบ VRAM ที่วัฒน์แล้ว

ไม่เขียน planner ใหม่: เรียก llama-fit-params ของ upstream ก่อน ถ้าใช้ไม่ได้จึงคิดเลขเอง
เลขที่คิดเองเป็น arithmetic ตรง ๆ จาก profile (ไม่ใช่ heuristic ลับ)

ใช้: python -m overspec.plan <model.gguf> [--ctx 4096] [--json]
"""
from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from . import budget as budget_mod
from .profile import Profile, read_profile

MIB = 1024 * 1024
GIB = 1024 ** 3
COMPUTE_OVERHEAD_MIB = 256      # scratch/compute buffer โดยประมาณ (วัฒน์จาก diag.log: 115 + 28 MiB)
DEFAULT_SAFETY_MIB = 768
DEFAULT_THREADS = 12            # ค่าที่วัฒน์ไว้ใน docs/plan.md:15
MOE_CACHE_SHARE = 0.10          # PR #29887 แนะนำ 10% ของขนาด expert

HERE = Path(__file__).resolve().parent
DEFAULT_SERVER = HERE.parent / "runtime" / "llama-server.exe"
DEFAULT_FIT = HERE.parent / "runtime-upstream" / "llama-fit-params.exe"

_ARGS_LINE = re.compile(r"^-(?:c|ngl|ot)\s", re.MULTILINE)
_FIT_C = re.compile(r"-c\s+(\d+)")
_FIT_NGL = re.compile(r"-ngl\s+(\d+)")


@dataclass
class Plan:
    argv: list[str]
    why: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    feasible: bool = True

    def as_dict(self) -> dict:
        return {
            "argv": self.argv,
            "why": self.why,
            "warnings": self.warnings,
            "feasible": self.feasible,
        }


def server_flags(server_bin: Path = DEFAULT_SERVER) -> set[str]:
    """flag ที่ build นี้รองรับ (อ่านจาก --help) — คืน set ว่างถ้าเรียกไม่ได้"""
    if not Path(server_bin).exists():
        return set()
    try:
        res = subprocess.run([str(server_bin), "--help"], capture_output=True,
                             text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return set()
    text = res.stdout + res.stderr
    found = set()
    for token in re.findall(r"--[a-z0-9][a-z0-9-]+", text):
        found.add(token)
    return found


def fit_params_args(model: str, fit_target_mib: int,
                    fit_binary: Path = DEFAULT_FIT) -> list[str] | None:
    """เรียก llama-fit-params แล้วดึง argv ที่มันแนะนำ (None = ใช้ไม่ได้)"""
    if not Path(fit_binary).exists():
        return None
    try:
        res = subprocess.run(
            [str(fit_binary), "--model", model, "-fitt", str(fit_target_mib), "-fa", "on"],
            capture_output=True, text=True, timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if res.returncode != 0:
        return None
    # บรรทัดสุดท้ายที่มี -c/-ngl คือ argv ที่ fit แนะนำ
    candidates = [ln.strip() for ln in (res.stdout + res.stderr).splitlines()
                  if _ARGS_LINE.search(ln)]
    if not candidates:
        return None
    try:
        return shlex.split(candidates[-1])
    except ValueError:
        return None


def _dense_fallback(profile: Profile, available_bytes: int) -> tuple[list[int], int]:
    per_layer = profile.bytes_per_layer()
    if not per_layer or not profile.layers:
        return [], 0
    mean = sum(per_layer) / profile.layers
    if mean <= 0:
        return [], 0
    n = max(0, min(profile.layers, int(available_bytes // mean)))
    return per_layer, n


def _ncmoe_for_budget(profile: Profile, expert_budget: int) -> int:
    """จำนวน layer แรกที่ต้องเก็บ expert ไว้บน CPU เพื่อให้ expert ที่เหลือพอดีงบ"""
    total = 0
    n = 0
    for layer in range(profile.layers - 1, -1, -1):
        size = profile.expert_bytes[layer]
        if total + size > expert_budget:
            break
        total += size
        n += 1
    return profile.layers - n


def make_plan(profile: Profile, budget_mib: int, ctx: int = 4096,
              safety_mib: int = DEFAULT_SAFETY_MIB,
              threads: int = DEFAULT_THREADS,
              server_bin: Path = DEFAULT_SERVER,
              fit_binary: Path = DEFAULT_FIT,
              use_fit: bool = True) -> Plan:
    kv_bytes = profile.kv_bytes_per_token * ctx
    available = (budget_mib - safety_mib) * MIB - kv_bytes - COMPUTE_OVERHEAD_MIB * MIB
    why = {
        "budget_mib": budget_mib,
        "safety_mib": safety_mib,
        "kv_mib": round(kv_bytes / MIB),
        "overhead_mib": COMPUTE_OVERHEAD_MIB,
        "available_mib": round(available / MIB),
        "threads": threads,
        "ctx": ctx,
    }
    warnings: list[str] = []
    flags = server_flags(server_bin)

    argv: list[str] = ["-c", str(ctx), "-fa", "on", "-t", str(threads)]

    if available <= 0:
        return Plan(argv, {**why, "source": "none"}, ["งบไม่พอแม้แต่ KV/overhead"], feasible=False)

    if profile.is_moe:
        dense = profile.total_dense_bytes
        experts = profile.total_expert_bytes
        fits_whole = dense + experts <= available
        if fits_whole:
            argv += ["-ngl", str(profile.layers + 2)]
            why["source"] = "arithmetic: ทั้งโมเดลพอดีงบ"
        elif "--moe-cache-mib" in flags:
            cache = int(min(experts * MOE_CACHE_SHARE, max(0, available - dense)))
            argv += ["-ngl", str(profile.layers + 2), "-cmoe", "--moe-cache-mib", str(cache // MIB)]
            why["source"] = "moe-cache"
            why["moe_cache_mib"] = cache // MIB
        else:
            expert_budget = max(0, available - dense)
            n = _ncmoe_for_budget(profile, expert_budget)
            argv += ["-ngl", str(profile.layers + 2), "-ncmoe", str(n)]
            why["source"] = "arithmetic: -ncmoe (build ไม่มี --moe-cache-mib)"
            why["ncmoe"] = n
            warnings.append("build ที่ใช้ไม่มี --moe-cache-mib — กำไรจาก expert cache จะไม่ได้")
        why["dense_mib"] = round(dense / MIB)
        why["expert_mib"] = round(experts / MIB)
        return Plan(argv, why, warnings)

    # dense model: ให้ upstream คิดก่อน
    if use_fit:
        fitted = fit_params_args(profile.path, max(1024, (budget_mib - safety_mib)))
        if fitted:
            m_c = _FIT_C.search(" ".join(fitted))
            m_ngl = _FIT_NGL.search(" ".join(fitted))
            argv = ["-c", m_c.group(1) if m_c else str(ctx), "-fa", "on", "-t", str(threads)]
            if m_ngl:
                argv += ["-ngl", m_ngl.group(1)]
            if "-ot" in fitted:
                idx = fitted.index("-ot")
                argv += ["-ot"] + fitted[idx + 1:idx + 2]
            why.update({"source": "llama-fit-params", "fit_argv": fitted})
            return Plan(argv, why, warnings)

    per_layer, n = _dense_fallback(profile, available)
    argv += ["-ngl", str(n)]
    why.update({
        "source": "arithmetic fallback",
        "n_gpu_layers": n,
        "bytes_per_layer_mean_mib": round(sum(per_layer) / max(1, len(per_layer)) / MIB),
    })
    if n < profile.layers:
        warnings.append(f"วางได้ {n}/{profile.layers} layer — ที่เหลือคำนวณบน CPU")
    return Plan(argv, why, warnings)


def plan_for_model(model: str, ctx: int = 4096, budget_mib: int | None = None,
                   snapshot_obj=None, profile: Profile | None = None, **kwargs) -> Plan:
    prof = profile or read_profile(model)
    if budget_mib is None:
        snap = snapshot_obj or budget_mod.snapshot()
        budget_mib = snap.available_mib()
        extra = {"budget_source": snap.budget_source()}
    else:
        extra = {"budget_source": "given"}
    plan = make_plan(prof, budget_mib, ctx=ctx, **kwargs)
    plan.why.update(extra)
    return plan


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = [a for a in argv if not a.startswith("-")]
    if not args:
        print("ใช้: python -m overspec.plan <model.gguf> [--ctx N] [--budget-mib N] [--json]",
              file=sys.stderr)
        return 2
    ctx = 4096
    if "--ctx" in argv:
        ctx = int(argv[argv.index("--ctx") + 1])
    budget_mib = None
    if "--budget-mib" in argv:
        budget_mib = int(argv[argv.index("--budget-mib") + 1])
    plan = plan_for_model(args[0], ctx=ctx, budget_mib=budget_mib)
    if "--json" in argv:
        print(json.dumps(plan.as_dict(), ensure_ascii=False, indent=2))
    else:
        print(" ".join(plan.argv))
        print("เหตุผล:")
        for k, v in plan.why.items():
            print(f"  {k}: {v}")
        for w in plan.warnings:
            print(f"  เตือน: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
