"""overspec.registry — เก็บ config ที่วัฒน์แล้วพร้อมคะแนนคุณภาพ

config หนึ่งตัว = (โมเดล, argv ที่รันจริง, ผลวัฒน์, คะแนน eval)
สถานะ: candidate → production (ผ่าน gate) หรือ rejected

ใช้: python -m overspec.registry list|promote|reject
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "configs" / "registry.json"
KEY_LEN = 12
STATUSES = ("candidate", "production", "rejected")


def config_key(model: str, argv: list[str]) -> str:
    """คีย์คงที่ของ config — โมเดล + argv เรียงตามที่ใช้จริง"""
    payload = json.dumps({"model": Path(model).name, "argv": argv}, ensure_ascii=False,
                         sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:KEY_LEN]


@dataclass
class Record:
    key: str
    model: str
    argv: list[str]
    verdict: str = "unknown"
    spill_ratio: float | None = None
    allocated_mib: int | None = None
    resident_mib: int | None = None
    tok_per_sec: float | None = None
    eval_passed: int | None = None
    eval_total: int | None = None
    status: str = "candidate"
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "key": self.key, "model": Path(self.model).name, "argv": self.argv,
            "verdict": self.verdict, "spill_ratio": self.spill_ratio,
            "allocated_mib": self.allocated_mib, "resident_mib": self.resident_mib,
            "tok_per_sec": self.tok_per_sec, "eval_passed": self.eval_passed,
            "eval_total": self.eval_total, "status": self.status,
            "created_at": self.created_at, "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Record":
        status = d.get("status", "candidate")
        if status not in STATUSES:
            status = "candidate"
        return cls(
            key=d["key"], model=d.get("model", ""), argv=list(d.get("argv", [])),
            verdict=d.get("verdict", "unknown"), spill_ratio=d.get("spill_ratio"),
            allocated_mib=d.get("allocated_mib"), resident_mib=d.get("resident_mib"),
            tok_per_sec=d.get("tok_per_sec"), eval_passed=d.get("eval_passed"),
            eval_total=d.get("eval_total"), status=status,
            created_at=d.get("created_at", ""), notes=list(d.get("notes", [])),
        )

    @property
    def eval_ratio(self) -> float | None:
        if not self.eval_total:
            return None
        return (self.eval_passed or 0) / self.eval_total


class Registry:
    def __init__(self, path: Path | None = None):
        self.path = Path(path or DEFAULT_PATH)
        self._records: dict[str, Record] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self._records = {}
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self._records = {r["key"]: Record.from_dict(r) for r in data.get("records", [])}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1,
                   "records": [r.to_dict() for r in sorted(self._records.values(),
                                                           key=lambda r: r.created_at)]}
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")

    def add(self, record: Record) -> Record:
        """เพิ่มหรืออัปเดต record เดิม (คีย์เดียวกัน = config เดียวกัน)"""
        old = self._records.get(record.key)
        if old is not None:
            record.created_at = old.created_at
            if old.status == "production" and record.status == "candidate":
                record.status = "production"
        self._records[record.key] = record
        self.save()
        return record

    def get(self, key: str) -> Record | None:
        return self._records.get(key)

    def all(self) -> list[Record]:
        return sorted(self._records.values(), key=lambda r: r.created_at, reverse=True)

    def production(self, model: str | None = None) -> list[Record]:
        name = Path(model).name if model else None
        return [r for r in self._records.values()
                if r.status == "production" and (name is None or r.model == name)]

    def promote(self, key: str) -> Record:
        """เลื่อนเป็น production — โมเดลเดียวกันมี production ได้ตัวเดียว"""
        rec = self._require(key)
        for other in self._records.values():
            if other.model == rec.model and other.status == "production":
                other.status = "candidate"
        rec.status = "production"
        self.save()
        return rec

    def reject(self, key: str) -> Record:
        rec = self._require(key)
        rec.status = "rejected"
        self.save()
        return rec

    def _require(self, key: str) -> Record:
        rec = self._records.get(key)
        if rec is None:
            raise KeyError(f"ไม่พบ config คีย์ {key}")
        return rec


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print("ใช้: python -m overspec.registry list|promote <key>|reject <key>", file=sys.stderr)
        return 2
    reg = Registry()
    cmd = argv[0]
    if cmd == "list":
        rows = reg.all()
        if not rows:
            print("ยังไม่มี config ที่บันทึกไว้")
        for r in rows:
            ratio = f"{r.eval_passed}/{r.eval_total}" if r.eval_total else "-"
            print(f"{r.key}  {r.status:<10} {r.model:<28} {r.tok_per_sec or 0:6.2f} tok/s  "
                  f"spill={r.spill_ratio if r.spill_ratio is not None else '-'}  eval={ratio}")
        return 0
    if cmd in ("promote", "reject") and len(argv) > 1:
        rec = reg.promote(argv[1]) if cmd == "promote" else reg.reject(argv[1])
        print(f"{rec.key} → {rec.status}")
        return 0
    print(f"คำสั่งไม่รู้จัก: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
