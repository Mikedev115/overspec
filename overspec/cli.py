"""overspec.cli — คำสั่งเดียวของ OverSpec

python -m overspec.cli budget [--human]
python -m overspec.cli profile <model.gguf> [--json]
python -m overspec.cli plan <model.gguf> [--ctx N] [--budget-mib N] [--json]
python -m overspec.cli run <model.gguf> [--ctx N] [--port N] [--attempts N] [--json]
python -m overspec.cli eval --url http://127.0.0.1:8099 [--key KEY]
python -m overspec.cli registry list|promote KEY|reject KEY
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import budget as budget_mod
from . import eval as eval_mod
from . import plan as plan_mod
from . import profile as profile_mod
from . import registry as registry_mod
from . import supervise as supervise_mod


def _print(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def cmd_budget(args) -> int:
    snap = budget_mod.snapshot()
    if args.human:
        print(budget_mod.render_human(snap))
    else:
        _print(snap.as_dict())
    return 0


def _load_profile(model: str):
    """อ่าน GGUF — คืน (profile, None) หรือ (None, รหัสออก)"""
    try:
        return profile_mod.read_profile(model), None
    except FileNotFoundError:
        print(f"ไม่พบไฟล์โมเดล: {model}", file=sys.stderr)
        return None, 2
    except ValueError as exc:
        print(f"อ่านโมเดลไม่ได้: {exc}", file=sys.stderr)
        return None, 2


def cmd_profile(args) -> int:
    prof, code = _load_profile(args.model)
    if prof is None:
        return code
    if args.json:
        _print(prof.as_dict())
        return 0
    gib = 1024 ** 3
    print(f"{Path(args.model).name}: {prof.file_bytes / gib:.2f} GiB · {prof.layers} layers · "
          f"{prof.params_b:.2f} B · {prof.bpw:.2f} bpw")
    print(f"dense {prof.total_dense_bytes / gib:.2f} GiB · expert {prof.total_expert_bytes / gib:.2f} GiB"
          + (" (MoE)" if prof.is_moe else ""))
    print(f"KV/token {prof.kv_bytes_per_token / 1024:.1f} KiB (f16)")
    return 0


def cmd_plan(args) -> int:
    prof, code = _load_profile(args.model)
    if prof is None:
        return code
    plan = plan_mod.plan_for_model(args.model, ctx=args.ctx, budget_mib=args.budget_mib,
                                   profile=prof)
    if args.json:
        _print(plan.as_dict())
        return 0
    print(" ".join(plan.argv))
    print(f"ที่มา: {plan.why.get('source')}")
    for w in plan.warnings:
        print(f"เตือน: {w}")
    return 0 if plan.feasible else 1


def cmd_run(args) -> int:
    prof, code = _load_profile(args.model)
    if prof is None:
        return code
    snap = budget_mod.snapshot()
    budget_mib = args.budget_mib or snap.available_mib()
    results = supervise_mod.run_with_ladder(
        prof, budget_mib, args.model, ctx=args.ctx, attempts=args.attempts, port=args.port)
    last = results[-1]

    key = registry_mod.config_key(args.model, last.argv)
    rec = registry_mod.Record(
        key=key, model=args.model, argv=last.argv, verdict=last.verdict,
        spill_ratio=last.spill_ratio, allocated_mib=last.allocated_mib,
        resident_mib=last.resident_mib, tok_per_sec=last.tok_per_sec,
        notes=list(last.notes) + [f"budget_source={snap.budget_source()}",
                                  f"budget_mib={budget_mib}"],
    )
    registry_mod.Registry().add(rec)

    if args.json:
        _print({"key": key, "runs": [r.as_dict() for r in results]})
    else:
        print(f"ผล: {last.verdict} · spill={last.spill_ratio} · {last.tok_per_sec or 0:.2f} tok/s")
        print(f"config key: {key}")
        print(f"log: {last.log_path}")
        for note in last.notes:
            print(f"  {note}")
    return 0 if last.verdict == "ok" else 1


def cmd_eval(args) -> int:
    try:
        url = eval_mod.local_url(args.url)
    except ValueError as exc:
        print(f"URL ไม่ถูกต้อง: {exc}", file=sys.stderr)
        return 2
    report = eval_mod.run_eval(url)
    reg = registry_mod.Registry()
    gate_info = None
    if args.key:
        rec = reg.get(args.key)
        if rec is None:
            print(f"ไม่พบ config คีย์ {args.key}", file=sys.stderr)
            return 2
        baseline = None
        prod = reg.production(rec.model)
        if prod and prod[0].eval_passed is not None and prod[0].key != rec.key:
            baseline = prod[0].eval_passed
        rec.eval_passed = report.passed
        rec.eval_total = report.total
        if baseline is not None:
            gate_info = eval_mod.gate(baseline, report.passed, args.max_regressions)
            if not gate_info["passed"]:
                rec.status = "rejected"
        reg.add(rec)

    if args.json:
        out = report.as_dict()
        if gate_info:
            out["gate"] = gate_info
        _print(out)
    else:
        print(f"ผ่าน {report.passed}/{report.total}")
        for tag, row in report.by_tag().items():
            print(f"  {tag}: {row['passed']}/{row['total']}")
        if gate_info:
            verdict = "ผ่าน" if gate_info["passed"] else "ไม่ผ่าน (ตกเกินกำหนด)"
            print(f"gate: {verdict} · baseline {gate_info['baseline']} → {gate_info['candidate']}")
    return 0 if report.passed == report.total else 1


def cmd_registry(args) -> int:
    reg = registry_mod.Registry()
    if args.action == "list":
        rows = reg.all()
        if not rows:
            print("ยังไม่มี config ที่บันทึกไว้")
        for r in rows:
            ratio = f"{r.eval_passed}/{r.eval_total}" if r.eval_total else "-"
            print(f"{r.key}  {r.status:<10} {r.model:<28} {r.tok_per_sec or 0:6.2f} tok/s  "
                  f"spill={r.spill_ratio if r.spill_ratio is not None else '-'}  eval={ratio}")
        return 0
    try:
        rec = reg.promote(args.key) if args.action == "promote" else reg.reject(args.key)
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"{rec.key} → {rec.status}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="overspec")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("budget", help="อ่านงบ VRAM จริง")
    p.add_argument("--human", action="store_true")
    p.set_defaults(func=cmd_budget)

    p = sub.add_parser("profile", help="อ่าน GGUF เป็นแผนที่หน่วยความจำ")
    p.add_argument("model")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_profile)

    p = sub.add_parser("plan", help="สร้าง argv ของ llama-server")
    p.add_argument("model")
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--budget-mib", type=int, default=None)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("run", help="รัน + วัฒน์ + ตรวจ spill + บันทึก config")
    p.add_argument("model")
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--port", type=int, default=8099)
    p.add_argument("--attempts", type=int, default=3)
    p.add_argument("--budget-mib", type=int, default=None)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("eval", help="รันชุดข้อสอบกับ server ที่เปิดอยู่")
    p.add_argument("--url", required=True)
    p.add_argument("--key", default=None)
    p.add_argument("--max-regressions", type=int, default=2)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("registry", help="ดู/เลื่อน/ปฏิเสธ config")
    p.add_argument("action", choices=["list", "promote", "reject"])
    p.add_argument("key", nargs="?", default="")
    p.set_defaults(func=cmd_registry)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "registry" and args.action != "list" and not args.key:
        parser.error("ต้องระบุ key")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
