"""overspec.gguf_probe — อ่านหัวไฟล์ GGUF แล้วรายงานสิ่งที่ต้องรู้ก่อนวางแผน

ใช้:  python overspec/gguf_probe.py <path-to.gguf>

อ่านเฉพาะ metadata + tensor index (ไม่แตะ weights) จึงเร็วมาก แม้ไฟล์จะ 17 GB
"""
from __future__ import annotations

import os
import struct
import sys
from collections import Counter

# ค่า enum ของ ggml_type -> (ชื่อ, bits ต่อ weight)
GGML_TYPES = {
    0: ("F32", 32.0), 1: ("F16", 16.0), 2: ("Q4_0", 4.5), 3: ("Q4_1", 5.0),
    6: ("Q5_0", 5.5), 7: ("Q5_1", 6.0), 8: ("Q8_0", 8.5), 9: ("Q8_1", 9.0),
    10: ("Q2_K", 2.625), 11: ("Q3_K", 3.4375), 12: ("Q4_K", 4.5), 13: ("Q5_K", 5.5),
    14: ("Q6_K", 6.5625), 15: ("Q8_K", 8.0), 16: ("IQ2_XXS", 2.0625),
    17: ("IQ2_XS", 2.3125), 18: ("IQ3_XXS", 3.0625), 19: ("IQ1_S", 1.5625),
    20: ("IQ4_NL", 4.5), 21: ("IQ3_S", 3.4375), 22: ("IQ2_S", 2.5),
    23: ("IQ4_XS", 4.25), 24: ("I8", 8.0), 25: ("I16", 16.0), 26: ("I32", 32.0),
    27: ("I64", 64.0), 28: ("F64", 64.0), 29: ("IQ1_M", 1.75), 30: ("BF16", 16.0),
    31: ("Q4_0_4_4", 4.5), 32: ("Q4_0_4_8", 4.5), 33: ("Q4_0_8_8", 4.5),
    34: ("TQ1_0", 1.6875), 35: ("TQ2_0", 2.0625),
}

SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
          6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
TYPE_STRING = 8
TYPE_ARRAY = 9


class Reader:
    def __init__(self, fh):
        self.fh = fh

    def read(self, n):
        b = self.fh.read(n)
        if len(b) != n:
            raise EOFError("ไฟล์จบก่อนกำหนด")
        return b

    def u32(self):
        return struct.unpack("<I", self.read(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.read(8))[0]

    def string(self):
        n = self.u64()
        return self.read(n).decode("utf-8", "replace")

    def value(self, vtype):
        if vtype in SCALAR:
            size = struct.calcsize(SCALAR[vtype])
            return struct.unpack(SCALAR[vtype], self.read(size))[0]
        if vtype == TYPE_STRING:
            return self.string()
        if vtype == TYPE_ARRAY:
            itype = self.u32()
            count = self.u64()
            if itype == TYPE_STRING:
                return [self.string() for _ in range(count)]
            if itype in SCALAR:
                size = struct.calcsize(SCALAR[itype])
                return list(struct.unpack("<" + SCALAR[itype][1] * count,
                                          self.read(size * count)))
            for _ in range(count):
                self.value(itype)
            return None
        raise ValueError(f"metadata type {vtype} ไม่รู้จัก")


def probe(path):
    out = {"path": path, "size_gb": os.path.getsize(path) / 1e9}
    with open(path, "rb") as fh:
        r = Reader(fh)
        magic = r.read(4)
        if magic != b"GGUF":
            raise SystemExit(f"ไม่ใช่ไฟล์ GGUF (เจอ {magic!r})")
        out["version"] = r.u32()
        n_tensors = r.u64()
        n_kv = r.u64()
        out["tensors"] = n_tensors

        meta = {}
        for _ in range(n_kv):
            key = r.string()
            vtype = r.u32()
            val = r.value(vtype)
            if isinstance(val, (int, float, str)) and len(str(val)) < 400:
                meta[key] = val
        out["meta"] = meta

        per_type = Counter()
        per_type_elems = Counter()
        for _ in range(n_tensors):
            r.string()
            n_dims = r.u32()
            dims = [r.u64() for _ in range(n_dims)]
            ttype = r.u32()
            r.u64()
            elems = 1
            for d in dims:
                elems *= d
            per_type[ttype] += 1
            per_type_elems[ttype] += elems

    arch = next((k.split(".")[0] for k in meta if k.endswith(".block_count")), "unknown")
    out["arch"] = arch
    out["block_count"] = meta.get(f"{arch}.block_count")
    out["head_count"] = meta.get(f"{arch}.attention.head_count")
    out["head_count_kv"] = meta.get(f"{arch}.attention.head_count_kv")
    out["context"] = meta.get(f"{arch}.context_length")
    out["embedding"] = meta.get(f"{arch}.embedding_length")

    total_elems = sum(per_type_elems.values())
    bits = sum(GGML_TYPES.get(t, ("?", 0.0))[1] * e for t, e in per_type_elems.items())
    out["avg_bits"] = bits / total_elems if total_elems else 0.0
    out["params_b"] = total_elems / 1e9

    rows = []
    for t, count in per_type.most_common():
        name, bpw = GGML_TYPES.get(t, (f"type{t}", 0.0))
        elems = per_type_elems[t]
        share = 100.0 * elems / total_elems if total_elems else 0.0
        rows.append((name, count, share, bpw, elems * bpw / 8 / 1e9))
    out["type_rows"] = rows
    return out


def main():
    if len(sys.argv) < 2:
        raise SystemExit("ใช้: python overspec/gguf_probe.py <path-to.gguf>")
    d = probe(sys.argv[1])

    print(f"ไฟล์         : {os.path.basename(d['path'])}")
    print(f"ขนาด         : {d['size_gb']:.2f} GB")
    print(f"GGUF version : {d['version']} · tensors {d['tensors']}")
    print(f"architecture : {d['arch']} · layers {d['block_count']} · "
          f"heads {d['head_count']} / kv {d['head_count_kv']}")
    print(f"context      : {d['context']} · embedding {d['embedding']}")
    print(f"params       : {d['params_b']:.2f} B")
    print(f"เฉลี่ย        : {d['avg_bits']:.2f} bits/weight")

    if d["block_count"]:
        per_layer_gb = d["size_gb"] / d["block_count"]
        fits = int(6.3 / per_layer_gb)
        print(f"ต่อ layer     : ~{per_layer_gb * 1000:.0f} MB "
              f"→ VRAM ใช้ได้ ~6.3 GB = ~{fits} layer จาก {d['block_count']}")

    print("\nองค์ประกอบต่อชนิด:")
    print(f"{'ชนิด':<10}{'tensor':>8}{'สัดส่วน':>10}{'bits':>8}{'ขนาด GB':>10}")
    for name, count, share, bpw, gb in d["type_rows"][:12]:
        print(f"{name:<10}{count:>8}{share:>9.1f}%{bpw:>8.2f}{gb:>10.2f}")


if __name__ == "__main__":
    main()
