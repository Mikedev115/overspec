# บทเรียน 01 — พูดเรื่อง LLM runtime ให้รู้เรื่อง

> ต่อจาก `00-พื้นฐาน-การโหลดโมเดล.md` · ตัวเลขทุกตัวในนี้วัฒน์บนเครื่องเราเอง ไม่ใช่ค่าจากเว็บ

**ถ้าจำได้แค่ 6 บรรทัด**
1. ขนาดโมเดล = params × bpw ÷ 8 (9.2B × 4.93 bpw ≈ 5.28 GiB)
2. `decode` ติด bandwidth, `prefill` ติด compute และ page fault
3. "อยู่บนการ์ด" ต้องพิสูจน์ด้วย VRAM ที่ **เพิ่มจริง** ไม่ใช่ log ที่บอกว่า allocate เท่าไร
4. offload = ย้าย layer หรือ expert ลง CPU → ช้าลงตามแบนด์วิดท์ RAM
5. quant ต่ำลง = ไฟล์เล็กลง แต่คุณภาพตก ต้อง **วัด** ไม่ใช่เดา
6. ทุกตัวเลขต้องบอกได้ว่า "วัดตอนไหน เครื่องว่างไหม"

---

## 1. ไฟล์ GGUF ข้างในมีอะไร

| ส่วน | เก็บอะไร | ตัวอย่างของเรา |
|---|---|---|
| metadata | architecture, `block_count` (= จำนวน layer), `head_count`, `context_length`, alignment | `qwen35`, 32 blocks, ctx train 262144 |
| tensor index | ชื่อ tensor + dims + ชนิด quant + offset | 442 tensors ในไฟล์ 9B |
| weights | ตัวเลขจริงของโมเดล | 5.28 GiB (Q4_K - Medium, 4.93 bpw) |

อ่านเองได้ด้วย `python overspec/gguf_probe.py <file.gguf>` หรือ `python -m overspec.cli profile <file.gguf>`

**สูตรที่ใช้บ่อย**
```
ขนาดไฟล์ (GB) ≈ params(B) × bpw ÷ 8
9.2 × 4.93 ÷ 8 = 5.67 GB  ≈ 5.28 GiB  ✓
ต่อ layer = ขนาดไฟล์ ÷ layers
5.28 GiB ÷ 32 = 169 MB/layer
```

## 2. คำ 7 คำที่ต้องใช้ให้ถูก

| คำ | ความหมาย | ของเครื่องเรา |
|---|---|---|
| **layer** (blk) | ชั้นของ transformer | 9B = 32 · 27B = 65 |
| **bpw** | bits per weight เฉลี่ย (ยิ่งต่ำ ไฟล์ยิ่งเล็ก) | Q4_K_M = 4.93 |
| **n_ctx** | ขนาด context (หน่วยเป็น token) | ใช้ 4096 |
| **KV cache** | หน่วยความจำของ attention โตตาม ctx | 128 MiB ที่ ctx 4096 (f16) |
| **prefill** | อ่าน prompt ทั้งก้อนรอบเดียว | 9B = 2.6 tok/s (ผิดปกติ) |
| **decode** | ออก token ทีละตัว | 9B = 19.77–21.85 tok/s |
| **offload** | เอา layer/expert ไปไว้ CPU แทนการ์ด | `-ngl 32` |

## 3. ทำไมคำว่า "วางทั้งโมเดลบนการ์ด" ถึงหลอกตา (แกนของเรื่อง)

Windows ใช้ WDDM ซึ่ง**ให้จอง VRAM เกินจริงได้** แล้วสลับหน้าลง RAM เงียบ ๆ ผลคือ:

| วัด | ค่า |
|---|---|
| llama.cpp คิดว่าใช้บนการ์ด | 4458 MiB |
| VRAM เพิ่มจริงตอนโหลด | **266 MiB** |
| หายไปไหน | ~4192 MiB ถูกสลับลง RAM |

⇒ ทุกเครื่องมือจะรายงาน "on GPU 100%" ทั้งที่ของอยู่บน RAM — นี่คือเหตุที่ตัวเลข tok/s ต่ำกว่าที่ควร

**สิ่งที่เอาไว้เช็ก**
```
spill_ratio = 1 − (VRAM ที่เพิ่มจริง ÷ buffer ที่จอง)
ถ้า > 0.10 = อย่าเชื่อว่า resident
```
- `nvidia-smi --query-gpu=memory.used,memory.reserved,memory.free`
- Windows: counter `GPU Process Memory` → VRAM แยกตามโปรเซส (`overspec/probe_budget.ps1`)
- DXGI `QueryVideoMemoryInfo` เป็นค่าที่ถูกต้องตามทฤษฎี แต่เครื่องเราคืน 0 (วัฒน์แล้ว) → ใช้ `free − reserved`

## 4. offload มี 3 แบบ อย่าสับสน

| แบบ | flag | ย้ายอะไร |
|---|---|---|
| ทั้ง layer | `-ngl N` | N layer แรก/หลัง ขึ้นการ์ด |
| เลือก tensor | `-ot "pattern=CPU"` | tensor ที่ตรง pattern |
| MoE expert | `-cmoe` / `-ncmoe N` / `--moe-cache-mib` | เฉพาะ weights ของ expert |

ผลต่างของความเร็วมาจาก bandwidth: การ์ด ~384 GB/s (สเปกใน `docs/00`) · RAM 42.9 GB/s (วัฒน์จากเส้น `-ngl` ของเราเอง)
⇒ layer ที่อยู่บน CPU แพงกว่า ~9 เท่าต่อไบต์

## 5. quant: ทำไม "ลดเป็น Q2" ไม่ใช่คำตอบ

| ระดับ | bpw | ขนาด 9B | คุณภาพ |
|---|---|---|---|
| Q4_K_M | 4.93 | 5.28 GiB | PPL 6.7 (อังกฤษ) |
| IQ2 | ~2.5 | ~3 GB | PPL 7.39 → **แย่ลง 11%** |

- ไม่มี "FP2" ใน GGUF — ที่มีคือ Q2_K, IQ2_XXS, IQ2_S ฯลฯ
- ตัวเลข PPL ข้างบนเป็น **ภาษาอังกฤษ** ยังไม่มีข้อมูลงานไทย/โค้ด
- กติกาโปรเจกต์: เปลี่ยน quant ได้ต่อเมื่อมี eval ยืนยันว่าไม่ตก (`docs/plan.md:83`)

## 6. ทำไม prompt ยาวแล้วค้าง

prefill 9B วัฒน์ได้ 2.6 tok/s, 27B = 1.4 tok/s ⇒ prompt 2,000 token = 24 นาที
สาเหตุไม่ใช่ GPU อ่อน แต่คือ **paging** (RAM เต็ม 98%) — แก้ที่การโหลด ไม่ใช่ที่การ์ด

## 7. ตัวเลขเครื่องผม (ตอบเพื่อนได้เลย)

| | ค่า | ที่มา |
|---|---|---|
| การ์ด | RTX 5050 Laptop · 8151 MiB (CUDA เห็น 8123) | `overspec/probe_budget.ps1`, `diag.log:4` |
| VRAM ที่แอปอื่นถือ | 6–7 GB (Chrome/Edge/Roblox) | `docs/plan.md:117` |
| CPU | i5-14450HX 6P+4E · threads 12 | `diag.log:6` |
| RAM bandwidth | 42.9 GB/s | `README.md:31` |
| 9B Q4_K_M | 5.28 GiB · 32 layers · 4.93 bpw | `diag.log:89` |
| 9B decode | 19.77 tok/s (`-ngl 32`) · 21.85 (`-ngl 99`) | `docs/plan.md:15`, `README.md:30` |
| 27B Q4 | 16.81 GB · 65 layers · 259 MB/layer | `docs/plan.md:16` |

## 8. ห้าเรื่องที่คนพูดผิดบ่อย

1. **"32 layer = 4 GB = ทั้งโมเดล"** — 4 GB นั้นคือ buffer บนการ์ดของ *บางส่วน* ไฟล์จริง 5.28 GiB
2. **"ollama บอก 100% GPU"** — ไม่ได้แปลว่า weights อยู่บนการ์ดจริง (WDDM สลับได้)
3. **"Q2 กับ Q4 พอ ๆ กัน"** — PPL ต่าง 11% และยังไม่วัดงานไทย
4. **"ซื้อการ์ดแรงขึ้นจะเร็วขึ้น"** — decode ติด bandwidth ไม่ใช่ FLOPS; การ์ดแรงเท่าเดิมแต่แบนด์วิดท์สูงขึ้นต่างหากที่ช่วย
5. **"layer เยอะ = โมเดลใหญ่"** — 9B มี 32, 27B มี 65; ขนาดมาจาก params × bpw ไม่ใช่จำนวน layer

## 9. คำถาม 5 ข้อที่ทำให้รู้ว่าคนที่คุยด้วยวัดจริงหรือพูดตามเว็บ

1. ตัวเลข tok/s นั้นเป็น **prefill หรือ decode**?
2. ตอนวัด **เครื่องว่างไหม** และ VRAM เพิ่มจริงเท่าไร (ไม่ใช่ log)
3. quant อะไร **bpw เท่าไร**
4. วัฒน์คุณภาพด้วยอะไร กี่ข้อ
5. `-ngl` เท่าไร, ctx เท่าไร, `-fa` เปิดไหม

## 10. แบบฝึก (ตอบในหัวได้ก็พอ)

1. โมเดล 7B ที่ 4 bpw ใช้พื้นที่ประมาณกี่ GB → *3.5 GB*
2. ถ้า `-ngl 32` แล้ว VRAM เพิ่มแค่ 300 MiB จาก buffer 4458 MiB เกิดอะไร → *spill ~93%*
3. ทำไม MoE ถึงเร็วกว่า dense ที่ขนาดไฟล์เท่ากัน → *อ่านเฉพาะ expert ที่ถูกเลือก ไม่ใช่ทั้งก้อน*
4. `-cmoe` ต่างจาก `-ncmoe 10` อย่างไร → *ทั้งโมเดล vs 10 layer แรก*
5. ถ้า KT ต้องการ prompt 8,000 token ควรกังวลเรื่องใดก่อน → *prefill/paging ไม่ใช่ decode*
