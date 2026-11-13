"""overspec.eval — ประตูคุณภาพ: config ที่เร็วขึ้นแต่ตอบแย่ลงต้องไม่ถูกเลื่อนขึ้น production

ชุดข้อสอบตั้งต้นอยู่ใน overspec/eval/questions.jsonl (25 ข้อ: ไทย / โค้ด / tool-call)
แทนได้ด้วย subset ของ ThaiExam, ThaiLLM-benchmark หรือ BFCL เมื่อพร้อม

ใช้: python -m overspec.eval --url http://127.0.0.1:8099 --json
"""
from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
QUESTIONS = HERE / "eval_data" / "questions.jsonl"


@dataclass
class Item:
    id: str
    tag: str
    prompt: str
    expect: dict

    @classmethod
    def from_dict(cls, d: dict) -> "Item":
        return cls(id=d["id"], tag=d.get("tag", "other"), prompt=d["prompt"],
                   expect=d.get("expect", {}))


@dataclass
class EvalResult:
    item_id: str
    tag: str
    passed: bool
    answer: str
    error: str | None = None

    def as_dict(self) -> dict:
        return {"item_id": self.item_id, "tag": self.tag, "passed": self.passed,
                "answer": self.answer[:200], "error": self.error}


@dataclass
class EvalReport:
    results: list[EvalResult] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def total(self) -> int:
        return len(self.results)

    def by_tag(self) -> dict:
        out: dict[str, dict] = {}
        for r in self.results:
            row = out.setdefault(r.tag, {"passed": 0, "total": 0})
            row["total"] += 1
            if r.passed:
                row["passed"] += 1
        return out

    def as_dict(self) -> dict:
        return {"passed": self.passed, "total": self.total, "by_tag": self.by_tag(),
                "results": [r.as_dict() for r in self.results]}


def load_items(path: Path = QUESTIONS) -> list[Item]:
    items = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(Item.from_dict(json.loads(line)))
    return items


def score(answer: str, expect: dict) -> bool:
    """ตรวจคำตอบแบบ deterministic — exact / contains / regex

    pattern มาจากชุดข้อสอบของโปรเจกต์เอง ไม่ใช่จากผู้ใช้; ถ้าโหลดชุดอื่นต้องตรวจก่อนใช้
    """
    kind = expect.get("type", "regex")
    value = expect.get("value", "")
    text = answer.strip()
    if kind == "exact":
        return text == value
    if kind == "contains":
        return value in text
    if kind == "regex":
        if len(value) > 500:
            raise ValueError("pattern ยาวเกิน 500 ตัวอักษร")
        try:
            compiled = re.compile(value, re.IGNORECASE | re.DOTALL)
        except re.error as exc:
            raise ValueError(f"regex ในชุดข้อสอบไม่ถูกต้อง: {exc}") from exc
        return compiled.search(text) is not None
    raise ValueError(f"expect.type ไม่รู้จัก: {kind}")


LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def local_url(url: str) -> str:
    """รับเฉพาะ URL ของ server บนเครื่องตัวเอง — กันการยิง request ออกนอกเครื่อง"""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"scheme ไม่รองรับ: {parsed.scheme!r}")
    if (parsed.hostname or "") not in LOOPBACK_HOSTS:
        raise ValueError(f"อนุญาตเฉพาะ loopback (ได้ {parsed.hostname!r})")
    return url.rstrip("/")


def ask(url: str, prompt: str, n_predict: int = 96, timeout: int = 180) -> str:
    body = json.dumps({"prompt": prompt, "n_predict": n_predict, "temperature": 0}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as res:
        data = json.load(res)
    return str(data.get("content", ""))


def run_eval(url: str, items: list[Item] | None = None) -> EvalReport:
    report = EvalReport()
    for item in items if items is not None else load_items():
        try:
            answer = ask(url, item.prompt)
            ok = score(answer, item.expect)
            report.results.append(EvalResult(item.id, item.tag, ok, answer))
        except (urllib.error.URLError, OSError, ValueError) as exc:
            report.results.append(EvalResult(item.id, item.tag, False, "", str(exc)))
    return report


def gate(baseline: int, candidate: int, max_regressions: int = 2) -> dict:
    """เทียบจำนวนข้อที่ผ่าน — ตกเกิน max_regressions = ไม่ผ่าน"""
    delta = candidate - baseline
    return {
        "baseline": baseline,
        "candidate": candidate,
        "delta": delta,
        "passed": delta >= -abs(max_regressions),
        "max_regressions": max_regressions,
    }


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--url" not in argv:
        print("ใช้: python -m overspec.eval --url http://127.0.0.1:8099 [--json]", file=sys.stderr)
        return 2
    url = argv[argv.index("--url") + 1]
    report = run_eval(url)
    if "--json" in argv:
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    else:
        print(f"ผ่าน {report.passed}/{report.total}")
        for tag, row in report.by_tag().items():
            print(f"  {tag}: {row['passed']}/{row['total']}")
        for r in report.results:
            if not r.passed:
                print(f"  ตก: {r.item_id} — {r.error or r.answer[:80]!r}")
    return 0 if report.passed == report.total else 1


if __name__ == "__main__":
    raise SystemExit(main())
