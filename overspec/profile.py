"""overspec.profile — แปลงหัวไฟล์ GGUF เป็นแผนที่หน่วยความจำที่ planner ต้องใช้

ต่างจาก gguf_probe.py ที่รายงานภาพรวม: โมดูลนี้แยก **dense กับ expert ต่อ layer**
(เพราะ MoE วางสองส่วนนี้คนละที่ได้) และคำนวณ KV ต่อ token

ใช้: python -m overspec.profile <model.gguf> [--json]
"""
from __future__ import annotations

import json
import os
import re
import struct
import sys
from dataclasses import dataclass, field

LAYER_RE = re.compile(r"^blk\.(\d+)\.")
EXPERT_RE = re.compile(r"\.ffn_(?:gate|up|down)_exps\.")

TYPE_STRING = 8
TYPE_ARRAY = 9
SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
          6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


class _Reader:
    def __init__(self, fh):
        self.fh = fh
        self.pos = 0

    def read(self, n: int) -> bytes:
        b = self.fh.read(n)
        if len(b) != n:
            raise EOFError("ไฟล์จบก่อนกำหนด")
        self.pos += n
        return b

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def string(self) -> str:
        return self.read(self.u64()).decode("utf-8", "replace")

    def value(self, vtype: int):
        if vtype in SCALAR:
            return struct.unpack(SCALAR[vtype], self.read(struct.calcsize(SCALAR[vtype])))[0]
        if vtype == TYPE_STRING:
            return self.string()
        if vtype == TYPE_ARRAY:
            itype = self.u32()
            count = self.u64()
            if itype == TYPE_STRING:
                return [self.string() for _ in range(count)]
            if itype in SCALAR:
                size = struct.calcsize(SCALAR[itype])
                return list(struct.unpack("<" + SCALAR[itype][1] * count, self.read(size * count)))
            for _ in range(count):
                self.value(itype)
            return None
        raise ValueError(f"metadata type {vtype} ไม่รู้จัก")


@dataclass
class Profile:
    path: str
    arch: str = "unknown"
    layers: int = 0
    params_b: float = 0.0
    file_bytes: int = 0
    bpw: float = 0.0
    dense_bytes: list[int] = field(default_factory=list)
    expert_bytes: list[int] = field(default_factory=list)
    kv_bytes_per_token: int = 0
    tensors: int = 0
    head_count: int | None = None
    head_count_kv: int | None = None
    ctx_train: int | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def total_dense_bytes(self) -> int:
        return sum(self.dense_bytes)

    @property
    def total_expert_bytes(self) -> int:
        return sum(self.expert_bytes)

    @property
    def is_moe(self) -> bool:
        return self.total_expert_bytes > 0

    def bytes_per_layer(self) -> list[int]:
        return [d + e for d, e in zip(self.dense_bytes, self.expert_bytes)]

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "arch": self.arch,
            "layers": self.layers,
            "params_b": round(self.params_b, 3),
            "file_bytes": self.file_bytes,
            "file_gib": round(self.file_bytes / (1024 ** 3), 3),
            "bpw": round(self.bpw, 3),
            "tensors": self.tensors,
            "is_moe": self.is_moe,
            "total_dense_bytes": self.total_dense_bytes,
            "total_expert_bytes": self.total_expert_bytes,
            "dense_bytes_per_layer": self.dense_bytes,
            "expert_bytes_per_layer": self.expert_bytes,
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "head_count": self.head_count,
            "head_count_kv": self.head_count_kv,
            "ctx_train": self.ctx_train,
            "notes": self.notes,
        }


def _align(n: int, a: int) -> int:
    return (n + a - 1) // a * a


def _kv_bytes(meta: dict, arch: str, layers: int, notes: list[str]) -> int:
    head_count = meta.get(f"{arch}.attention.head_count")
    head_count_kv = meta.get(f"{arch}.attention.head_count_kv")
    embd = meta.get(f"{arch}.embedding_length")
    if not head_count or not embd or not layers:
        notes.append("ไม่มี metadata head_count/embedding_length — KV ต่อ token = 0")
        return 0

    def as_list(v, n):
        if isinstance(v, list):
            return [int(x) for x in v]
        return [int(v)] * n

    kv_list = as_list(head_count_kv if head_count_kv else head_count, layers)
    k_len = meta.get(f"{arch}.attention.key_length")
    v_len = meta.get(f"{arch}.attention.value_length")
    head_dim = int(k_len) if k_len else int(embd) // int(head_count)
    v_dim = int(v_len) if v_len else head_dim
    # เก็บ KV เป็น f16 (2 ไบต์) — dtype อื่นปรับภายหลังด้วย --cache-type-k/v
    return sum(h * (head_dim + v_dim) * 2 for h in kv_list[:layers])


def read_profile(path: str) -> Profile:
    size = os.path.getsize(path)
    prof = Profile(path=path, file_bytes=size)
    notes: list[str] = []

    with open(path, "rb") as fh:
        r = _Reader(fh)
        magic = r.read(4)
        if magic != b"GGUF":
            raise ValueError(f"ไม่ใช่ไฟล์ GGUF (เจอ {magic!r})")
        r.u32()  # version
        n_tensors = r.u64()
        n_kv = r.u64()

        meta: dict = {}
        for _ in range(n_kv):
            key = r.string()
            vtype = r.u32()
            val = r.value(vtype)
            if isinstance(val, (int, float, str)):
                meta[key] = val

        infos = []
        for _ in range(n_tensors):
            name = r.string()
            n_dims = r.u32()
            dims = [r.u64() for _ in range(n_dims)]
            ttype = r.u32()
            offset = r.u64()
            infos.append((name, dims, ttype, offset))

        alignment = int(meta.get("general.alignment", 32) or 32)
        data_start = _align(r.pos, alignment)

    arch = next((k.split(".")[0] for k in meta if k.endswith(".block_count")), "unknown")
    layers = int(meta.get(f"{arch}.block_count", 0) or 0)
    prof.arch = arch
    prof.layers = layers
    prof.tensors = len(infos)
    prof.head_count = meta.get(f"{arch}.attention.head_count")
    prof.head_count_kv = meta.get(f"{arch}.attention.head_count_kv")
    prof.ctx_train = meta.get(f"{arch}.context_length")

    # ขนาดจริงต่อ tensor: ใช้ส่วนต่างของ offset — ไม่ต้องเดา block size ของ quant
    ordered = sorted(infos, key=lambda t: t[3])
    sizes: dict[str, int] = {}
    for i, (name, dims, ttype, offset) in enumerate(ordered):
        if i + 1 < len(ordered):
            nbytes = ordered[i + 1][3] - offset
        else:
            nbytes = max(0, size - data_start - offset)
        sizes[name] = nbytes

    prof.dense_bytes = [0] * layers
    prof.expert_bytes = [0] * layers
    elems_total = 0
    for name, dims, ttype, _ in infos:
        n = 1
        for d in dims:
            n *= d
        elems_total += n
        m = LAYER_RE.match(name)
        if not m:
            continue
        layer = int(m.group(1))
        if layer >= layers:
            continue
        if EXPERT_RE.search(name):
            prof.expert_bytes[layer] += sizes.get(name, 0)
        else:
            prof.dense_bytes[layer] += sizes.get(name, 0)

    prof.params_b = elems_total / 1e9
    prof.bpw = (sum(sizes.values()) * 8 / elems_total) if elems_total else 0.0
    prof.kv_bytes_per_token = _kv_bytes(meta, arch, layers, notes)
    if prof.total_expert_bytes:
        notes.append(f"MoE: expert รวม {prof.total_expert_bytes / 1024 ** 3:.2f} GiB")
    prof.notes = notes
    return prof


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = [a for a in argv if not a.startswith("-")]
    if not args:
        print("ใช้: python -m overspec.profile <model.gguf> [--json]", file=sys.stderr)
        return 2
    prof = read_profile(args[0])
    if "--json" in argv:
        print(json.dumps(prof.as_dict(), ensure_ascii=False, indent=2))
        return 0

    gib = 1024 ** 3
    print(f"ไฟล์        : {os.path.basename(prof.path)} · {prof.file_bytes / gib:.2f} GiB · {prof.bpw:.2f} bpw")
    print(f"arch        : {prof.arch} · {prof.layers} layers · {prof.params_b:.2f} B params · {prof.tensors} tensors")
    print(f"dense       : {prof.total_dense_bytes / gib:.2f} GiB · expert: {prof.total_expert_bytes / gib:.2f} GiB"
          + ("  (MoE)" if prof.is_moe else ""))
    print(f"KV/token    : {prof.kv_bytes_per_token / 1024:.1f} KiB (f16) · ctx train {prof.ctx_train}")
    for line in prof.notes:
        print(f"หมายเหตุ    : {line}")
    per_layer = prof.bytes_per_layer()[:6]
    if per_layer:
        print("ต่อ layer   : " + ", ".join(f"{b / 1024 ** 2:.0f} MB" for b in per_layer) + " ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
