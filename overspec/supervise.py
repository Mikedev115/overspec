"""overspec.supervise — ตรวจว่า weights อยู่บนการ์ดจริงหลังโหลด และถอยขั้นบันไดเมื่อ spill

ปัญหาที่โมดูลนี้แก้: llama.cpp รายงาน "CUDA0 model buffer = 4458 MiB" แต่ VRAM เพิ่มจริง
แค่ 266 MiB เพราะ WDDM สลับลง RAM เงียบ ๆ (วัฒน์ใน README.md ข้อ "ค้นพบหลัก")

ใช้: python -m overspec.supervise --model X [--argv-from-plan]
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from . import budget as budget_mod
from .plan import DEFAULT_SERVER, Plan, make_plan
from .profile import Profile, read_profile

MIB = 1024 * 1024
ALLOCATED_RE = re.compile(r"CUDA0\s+model buffer size\s*=\s*([\d.]+)\s*MiB")
SPILL_THRESHOLD = 0.10
POLL_SECONDS = 1.0


@dataclass
class RunResult:
    verdict: str                  # ok | spill | failed
    argv: list[str] = field(default_factory=list)
    allocated_mib: int | None = None
    resident_mib: int | None = None
    spill_ratio: float | None = None
    tok_per_sec: float | None = None
    log_path: str | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "argv": self.argv,
            "allocated_mib": self.allocated_mib,
            "resident_mib": self.resident_mib,
            "spill_ratio": self.spill_ratio,
            "tok_per_sec": self.tok_per_sec,
            "log_path": self.log_path,
            "notes": self.notes,
        }


def parse_allocated_mib(log_text: str) -> int | None:
    """ดึง 'CUDA0 model buffer size = 4458.03 MiB'"""
    matches = ALLOCATED_RE.findall(log_text)
    if not matches:
        return None
    return int(round(float(matches[-1])))


def compute_spill_ratio(allocated_mib: int | None, resident_mib: int | None) -> float | None:
    """สัดส่วน weights ที่ไม่ได้อยู่บนการ์ดจริง (0 = อยู่ครบ)"""
    if not allocated_mib or resident_mib is None:
        return None
    if allocated_mib <= 0:
        return None
    ratio = 1 - (resident_mib / allocated_mib)
    return max(0.0, min(1.0, ratio))


def ladder(plan: Plan, attempt: int) -> Plan | None:
    """ขั้นถัดไปเมื่อเจอ spill: ลด ctx ครึ่ง → ลด layer 4 → ลด moe cache ครึ่ง

    คืน None เมื่อไม่มีขั้นให้ถอยแล้ว
    """
    argv = list(plan.argv)
    why = dict(plan.why)
    if attempt == 0:
        if "-c" in argv:
            i = argv.index("-c")
            new_ctx = max(2048, int(argv[i + 1]) // 2)
            if new_ctx >= int(argv[i + 1]):
                return None
            argv[i + 1] = str(new_ctx)
            why["ladder"] = "ลด ctx → " + str(new_ctx)
            return Plan(argv, why, list(plan.warnings))
    if attempt == 1:
        if "-ngl" in argv:
            i = argv.index("-ngl")
            new_ngl = int(argv[i + 1]) - 4
            if new_ngl <= 0:
                return None
            argv[i + 1] = str(new_ngl)
            why["ladder"] = "ลด -ngl → " + str(new_ngl)
            return Plan(argv, why, list(plan.warnings))
    if attempt == 2:
        if "--moe-cache-mib" in argv:
            i = argv.index("--moe-cache-mib")
            new_cache = int(argv[i + 1]) // 2
            if new_cache < 512:
                return None
            argv[i + 1] = str(new_cache)
            why["ladder"] = "ลด moe cache → " + str(new_cache)
            return Plan(argv, why, list(plan.warnings))
    return None


def _local_base(port: int) -> str:
    """URL สำหรับ server ที่เราสตาร์ทเอง — รับเฉพาะ loopback กับ port ที่ถูกต้อง"""
    if isinstance(port, bool) or not isinstance(port, int) or not (1 <= port <= 65535):
        raise ValueError(f"port ไม่ถูกต้อง: {port!r}")
    return f"http://127.0.0.1:{port}"


def _health(url: str) -> bool:
    try:
        with urllib.request.urlopen(url + "/health", timeout=2) as res:
            return res.status == 200
    except (urllib.error.URLError, OSError):
        return False


def _tok_s(url: str, n_predict: int = 64) -> float | None:
    body = json.dumps({"prompt": "อธิบายสั้น ๆ ว่า quantization คืออะไร",
                       "n_predict": n_predict, "temperature": 0}).encode()
    req = urllib.request.Request(url + "/completion", data=body,
                                headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as res:
            data = json.load(res)
        return float(data["timings"]["predicted_per_second"])
    except (urllib.error.URLError, OSError, KeyError, ValueError):
        return None


def build_cmd(server_bin: Path, model: str, port: int, plan_argv: list[str],
              server_cmd: list[str] | None = None) -> list[str]:
    """คำสั่งที่ใช้สตาร์ท — server_cmd ไว้ให้เทสต์หรือ build อื่นแทนได้ทั้งคำสั่ง"""
    if server_cmd:
        return list(server_cmd)
    return [str(server_bin), "-m", model, "--port", str(port), "--parallel", "1", *plan_argv]


def run_once(plan: Plan, model: str, server_bin: Path = DEFAULT_SERVER,
             port: int = 8099, startup_timeout: int = 180,
             workdir: Path | None = None,
             server_cmd: list[str] | None = None,
             snapshot_fn=budget_mod.snapshot) -> RunResult:
    """สตาร์ท server หนึ่งรอบ วัฒน์ แล้วปิด — คืนผลว่ามี spill ไหม"""
    workdir = workdir or Path.cwd()
    workdir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = workdir / f"server-{stamp}.log"
    url = _local_base(port)

    before = snapshot_fn()
    cmd = build_cmd(server_bin, model, port, plan.argv, server_cmd)
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)

    result = RunResult(verdict="failed", argv=plan.argv, log_path=str(log_path))
    try:
        deadline = time.time() + startup_timeout
        alive = False
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            if _health(url):
                alive = True
                break
            time.sleep(POLL_SECONDS)

        log_text = Path(log_path).read_text(encoding="utf-8", errors="replace")
        result.allocated_mib = parse_allocated_mib(log_text)
        if not alive:
            result.notes.append("server ไม่ขึ้นภายในเวลาที่กำหนด")
            return result

        after = snapshot_fn()
        result.resident_mib = after.our_process_mib(proc.pid)
        result.spill_ratio = compute_spill_ratio(result.allocated_mib, result.resident_mib)
        result.tok_per_sec = _tok_s(url)

        if result.spill_ratio is None:
            result.verdict = "ok"
            result.notes.append("อ่านค่า allocation/resident ไม่ได้ — ตัดสินแบบไม่ยืนยัน")
        elif result.spill_ratio > SPILL_THRESHOLD:
            result.verdict = "spill"
        else:
            result.verdict = "ok"
        return result
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


def run_with_ladder(profile: Profile, budget_mib: int, model: str, ctx: int = 4096,
                    attempts: int = 3, **run_kwargs) -> list[RunResult]:
    """วน plan → วัฒน์ → ถ้า spill ถอยขั้น ตาม ladder"""
    plan = make_plan(profile, budget_mib, ctx=ctx)
    results: list[RunResult] = []
    for attempt in range(attempts):
        res = run_once(plan, model, **run_kwargs)
        results.append(res)
        if res.verdict == "ok":
            break
        nxt = ladder(plan, attempt)
        if nxt is None:
            break
        plan = nxt
    return results


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = [a for a in argv if not a.startswith("-")]
    if not args:
        print("ใช้: python -m overspec.supervise <model.gguf> [--budget-mib N] [--port N] [--json]",
              file=sys.stderr)
        return 2
    model = args[0]
    budget_mib = int(argv[argv.index("--budget-mib") + 1]) if "--budget-mib" in argv else None
    port = int(argv[argv.index("--port") + 1]) if "--port" in argv else 8099
    prof = read_profile(model)
    if budget_mib is None:
        budget_mib = budget_mod.snapshot().available_mib()
    results = run_with_ladder(prof, budget_mib, model, port=port)
    last = results[-1]
    if "--json" in argv:
        print(json.dumps([r.as_dict() for r in results], ensure_ascii=False, indent=2))
    else:
        print(f"ผล: {last.verdict} · allocated={last.allocated_mib} MiB · "
              f"resident={last.resident_mib} MiB · spill={last.spill_ratio} · "
              f"{last.tok_per_sec or 0:.2f} tok/s")
        for note in last.notes:
            print(f"  {note}")
    return 0 if last.verdict == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
