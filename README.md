# OverSpec

> ทำงานเกินสเปกเครื่อง

ตัวกลางที่คุมการโหลด การวาง weights และงบหน่วยความจำ เพื่อให้โมเดลรันบน **RTX 5050 Laptop 8GB + i5-14450HX (6P+4E) + RAM 32GB DDR5-5600** ได้เร็วขึ้นและไม่ทำให้เครื่องค้าง

---

## สองสัญญา

| สัญญา | ค่าเป้า | สถานะ |
|---|---|---|
| เร็วขึ้น | +30% เหนือ baseline | 9B: **+67%** (13.11 → 21.85 tok/s) ✅ |
| เครื่องใช้งานได้ | RAM เหลือ ≥6 GB · ไม่มี paging | ยังไม่ผ่าน |

## สถานะ (13 พ.ย. 2026 02:50)

**มีแล้ว**
- `overspec/gguf_probe.py` — อ่านหัวไฟล์ GGUF จริง (layers, bits/weight, ขนาดต่อ layer)
- `runtime/` — llama.cpp + CUDA ที่ประกอบเอง เห็นการ์ด (`CUDA0: RTX 5050, 7070 MiB`)
- `docs/00-พื้นฐาน-การโหลดโมเดล.md` — บทเรียน + แบบฝึก 5 ข้อที่ทดสอบแล้ว
- `docs/plan.md` — แผน v2 (แทนที่ฉบับแรกทั้งหมด)
- `docs/research-prior-art.md` — งานวิจัย/ของเดิมที่ทำเรื่องนี้แล้ว และช่องว่างที่เหลือ
- `docs/architecture.md` — สถาปัตยกรรม v3 (ไม่สร้าง runtime เอง; 3 ชั้นบน llama.cpp)
- `docs/01-พื้นฐาน-llm-runtime.md` — บทเรียนคำศัพท์/กลไก สำหรับคุยกับคนทำด้านนี้
- `overspec/probe_budget.ps1` — อ่านงบ VRAM จริงของ Windows แยกตามโปรเซส

**ตัวเลขที่วัดจริง**

| | ค่า |
|---|---|
| Qwen3.8-27B Q4 ผ่าน ollama | 3.93 tok/s |
| qwen3.5:9b ผ่าน ollama | 13.11 tok/s |
| qwen3.5:9b ผ่าน llama.cpp `-ngl 99` | **21.85 tok/s** |
| bandwidth RAM | 42.9 GB/s |
| ไฟล์ 27B | 16.81 GB · 65 layers · 4.92 bits · 259 MB/layer |

## ค้นพบหลัก

**WDDM spill** — llama.cpp จอง 4458 MiB บนการ์ด แต่ VRAM เพิ่มจริงแค่ 266 MiB เพราะแอปของผู้ใช้ถืออยู่ 7119 MiB จาก 8151
⇒ weights ~4.2 GB ถูกสลับลง RAM เงียบ ๆ ทั้งที่ทุกเครื่องมือรายงานว่า "100% GPU"
นี่คือกำแพงจริง และเป็นช่องที่ OverSpec ต่างจากคนอื่น: **รายงานและบังคับ VRAM budget จริง**

## อะไรอยู่ในรีโปนี้

| อยู่ในรีโป | ไม่อยู่ |
|---|---|
| `docs/` · `overspec/gguf_probe.py` · `runtime/wsl-bench.sh` · ตัวเลขที่วัดแล้ว | ไบนารี `runtime/*.dll`/`*.exe` · `runtime-upstream/` (llama.cpp + CUDA build, zip) · log การรัน |

ไบนารีใหญ่หลาย GB และโหลดได้จาก [ggml-org/llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) — ตัวที่ใช้ในผลวัดคือ build ที่รองรับ GGUF ของ ollama (fork) ไม่ใช่ build ต้นทาง b12321 (ดู `docs/plan.md` ข้อ 2)

## กติกา

- **ไม่ลดคุณภาพของโมเดล** เพื่อแลกความเร็ว
- ไม่เขียน kernel เอง (CPU เป็น AVX2 ไม่มี AVX-512/AMX)
- ไม่ทำ streaming จาก SSD (7 GB/s < RAM 42.9 GB/s)
- ทุกตัวเลขต้องมาจากการวัดบนเครื่องนี้ ไม่ cite เว็บ

## ใช้งาน

```bash
python -m overspec.cli budget --human                  # งบ VRAM จริง + VRAM แยกโปรเซส
python -m overspec.cli profile model.gguf --json       # dense/expert ต่อ layer + KV ต่อ token
python -m overspec.cli plan model.gguf --ctx 4096      # argv ที่ควรใช้ พร้อมเหตุผล
python -m overspec.cli run model.gguf --ctx 4096       # รัน + วัฒน์ + ตรวจ spill + บันทึก config
python -m overspec.cli eval --url http://127.0.0.1:8099 --key <config-key>   # ประตูคุณภาพ 25 ข้อ
python -m overspec.cli registry list
python -m pytest -q                                    # 42 เทสต์ (ไม่ต้องมีโมเดล/การ์ด)
```

`run` จะวัฒน์ tok/s และเทียบ `CUDA0 model buffer` กับ VRAM ที่โปรเซสถือจริง ถ้า spill เกิน 10% จะไล่ลด ctx → layer → moe cache เอง
