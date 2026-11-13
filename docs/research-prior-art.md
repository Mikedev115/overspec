# งานวิจัยและของเดิมที่ทำเรื่องนี้แล้ว

> ตรวจเมื่อ 13 พ.ย. 2026 · ทุกข้ออ้างมีลิงก์หรือไฟล์ในเครื่องกำกับ · ข้อที่ยังไม่ได้ยืนยันจะเขียนว่า "ยังไม่ยืนยัน"

**สรุปคำตอบ:** ปัญหา "วาง weights ให้พอดี VRAM" กับ "offload expert ของ MoE" มีคนทำจบแล้วทั้งสองชั้น — ชั้นที่ยังว่างจริงมี 3 ข้อ (งบ VRAM จริงบน Windows · การยืนยันว่า spill เกิดขึ้นจริง · การผูกคุณภาพเข้ากับการวาง) การสร้าง placement/runtime ใหม่เองคือการทำซ้ำ

---

## 1. ทำแล้ว / ยังว่าง

| ชั้น | สถานะ | ของที่มีอยู่ | หลักฐาน |
|---|---|---|---|
| วาง layer ให้พอดี VRAM อัตโนมัติ | **ทำแล้ว** | llama.cpp `--fit` / `llama_params_fit` | [tools/fit-params](https://github.com/ggml-org/llama.cpp/tree/master/tools/fit-params) |
| ลด context อัตโนมัติเมื่อไม่พอ | **ทำแล้ว** | `--fit-ctx`, `--fit-target` | `llama-server --help` (build ในเครื่อง) |
| วาง expert ของ MoE ลง CPU | **ทำแล้ว** | `-cmoe`, `-ncmoe N`, `-ncffn N`, `-ot regex` | [common/arg.cpp](https://github.com/ggml-org/llama.cpp/blob/master/common/arg.cpp) |
| cache expert บนการ์ด (LRU) | **ทำแล้ว** (mainline) | `--moe-cache-mib N`, `--moe-cache-cpu-share F` | [PR #29887](https://github.com/ggml-org/llama.cpp/pull/29887) merged 7 ต.ค. 2026 |
| โหลด/เก็บ KV, prompt cache | **ทำแล้ว** | `-ctk/-ctv`, `-kvu`, `--cache-ram`, `--ctx-checkpoints` | `arg.cpp`, `--help` |
| สลับโมเดล / route ตาม request | **ทำแล้ว** | llama-swap, SmarterRouter | [llama-swap](https://betterstack.com/community/guides/ai/llama-swap/), [SmarterRouter](https://insiderllm.com/guides/smarterrouter-vram-aware-llm-gateway-local-ai/) |
| **งบ VRAM จริงแบบ WDDM** | **ว่าง** | ไม่มีใครใน llama.cpp อ่าน DXGI Budget | §4 |
| **ยืนยันว่า weights resident จริง** | **ว่าง** | ไม่มีเครื่องมือไหนเทียบ buffer ที่จองกับ VRAM ที่เพิ่มจริง | §4, `README.md:36-38` |
| **ผูกคุณภาพเข้ากับ config ที่วาง** | **ว่าง** | มีชุดข้อสอบ แต่ไม่มีใครต่อเข้ากับการวาง | §5 |

---

## 2. llama.cpp: ทำอะไรได้แล้วบ้าง (ตรวจจาก source ไม่ใช่บล็อก)

### 2.1 fit params — ตัววางแผนที่ upstream มีอยู่แล้ว

`llama-fit-params` พิมพ์ argument ที่ควรใช้ (`-c`, `-ngl`, `-ot`) ออกมาเป็น stdout หลังคำนวณจากหน่วยความจำว่างจริง ตัวอย่างใน README ต้นทาง: context 40960 → 4096, `-ngl 48`, จ่าย 19187 MiB จาก 4090

ใน `common/common.h` (master):

```cpp
bool    fit_params         = true;  // whether to fit unset model/context parameters to free device memory
int32_t fit_params_min_ctx = 4096;  // minimum context size to set when trying to reduce memory use
std::vector<size_t> fit_params_target = ... 1024*1024*1024;  // margin per device
```

⇒ **สิ่งที่ upstream ยังไม่ทำ:** `fit_params` ใช้ "free device memory" จาก backend เป็นเพดาน ไม่ใช่งบของ OS และ **ไม่รู้ว่าแอปอื่นถือ VRAM อยู่เท่าไร** — ดูหลักฐานในเครื่อง: `runtime/diag.log:22-25` ("projected to use 4747 MiB ... vs. 2473 MiB of free device memory" แล้ว abort เพราะผู้ใช้ตั้ง `-ngl` เอง)

### 2.2 MoE expert cache — ของใหม่ที่ build ในเครื่องยังไม่มี

[PR #29887](https://github.com/ggml-org/llama.cpp/pull/29887) (merged 7 ต.ค. 2026, ผู้เขียน am17an, port จาก qvac-fabric):

- `MUL_MAT_ID` ของ expert ที่อยู่บน host รันบน GPU ผ่าน LRU cache, อัปโหลดเฉพาะตัวที่ miss
- ใช้ได้กับ batch เล็ก (≤ 32 tokens), single device เท่านั้น; layer ที่ layout ต่างกันแยก bank กัน
- **ขนาดที่แนะนำ = 10% ของขนาด expert ทั้งหมด**
- `--fit` นับ cache นี้ด้วย (จึงเหลือ whole expert layer บนการ์ดน้อยลง)

ตัวเลขจากเจ้าของ PR (Qwen3.8-Flash-Next Q4_0, 65.4 GiB experts, `--osl 512`, temp 0):

| การ์ด | master `--fit` | + `--moe-cache-mib` | เร็วขึ้น | hit rate |
|---|---|---|---|---|
| RTX 4090 | 25.0 tok/s | 39.4 tok/s (`6544`) | 1.57× | 72% |
| RTX 4090 | — | 40.7 tok/s (`-cmoe` + `11000`) | 1.62× | 77% |
| RTX 5090 | 30.8 tok/s | 54.5 tok/s (`6544`) | 1.77× | 77% |
| RTX 5090 | — | 67.8 tok/s (`-cmoe` + `19000`) | 2.20× | 89% |

**สถานะในเครื่องเรา:** build ที่ใช้อยู่คือ `0.3.0-dev (build 1, commit d222767c7)` ของ fork ollama — `--help` **ไม่มี** `--moe-cache-mib` / `--moe-cache-cpu-share` / `-ncffn` (grep แล้วไม่พบ) แต่มี `--fit`, `-fitt`, `-fitc`, `-cmoe`, `-ncmoe`, `-ot`, `-lm/--load-mode`, `-cram`, `-ctxcp`, `-kvu`

### 2.3 ข้อที่บล็อกอ้างแต่ source ไม่ตรง (กันการทำซ้ำผิดทาง)

- บล็อกหลายแห่งอ้าง PR #29191 ว่ามี `--load-mode streaming` — `arg.cpp` master วันนี้มีค่าแค่ `auto | none | mmap | mlock | mmap+mlock | dio` ⇒ **ไม่มี streaming** อย่าเอาไปออกแบบต่อ
- ที่มีจริงคือ `-lzm/--lazy-mode` (อ่านบาง tensor จากดิสก์ตามต้องการ ต้องใช้ mmap, auto เมื่อ tensor > 4 GiB)

---

## 3. งานวิจัย MoE offload — ทำถึงไหนแล้ว

| ระบบ | แนวคิด | ทำไมไม่สร้างใหม่ |
|---|---|---|
| FlexGen (SOSP'23) | LP หา placement ของ weights/KV บน GPU-CPU-disk เพื่อ throughput | เป้าเป็น throughput/offline; เราเป้า latency single-user — llama.cpp `--fit` ครอบแล้ว |
| ZeRO-Inference | เก็บ weights ใน CPU/NVMe แล้ว stream ทีละ layer | RAM 32 GB ทำให้ 9B อยู่ครบ RAM ได้ ไม่ต้อง stream (และ SSD 7 GB/s < RAM 42.9 GB/s — `README.md:44`) |
| PowerInfer (SOSP'24) | แยก hot/cold neuron, preload hot | ต้อง profile/เทรนเฉพาะโมเดล + เสี่ยงคุณภาพ ⇒ ขัดกติกา "ไม่ลดคุณภาพ" |
| Fiddler (ICLR'25, [efeslab](https://github.com/efeslab/fiddler)) | ให้ **CPU คำนวณ** expert แทนการย้ายข้อมูล | เท่ากับ `-cmoe`/`-ncmoe` ของ llama.cpp; และ CPU เราเป็น AVX2 ไม่มี AVX-512/AMX (`docs/plan.md:31`) |
| MoE-Infinity | expert cache ที่รู้ sparsity บนเครื่องส่วนตัว | ความคิดเดียวกับ `--moe-cache-mib` ที่ merged แล้ว |
| FloE (ICML'25) | ย้าย expert แบบ on-the-fly พร้อม pipeline | ต้องแก้ engine; ของ upstream ทำ LRU cache แล้ว |
| MoBiLE ([arXiv 2510.12357](https://arxiv.org/abs/2510.12357)) | ลดจำนวน expert ครึ่งหนึ่งสำหรับ token ที่สำคัญน้อย | **ลดคุณภาพ** ⇒ ขัดกติกาโครงการ |
| SMOE | dynamic expert cache แบบ CPU/GPU hybrid | ซ้ำกับ `--moe-cache-mib` |
| NEO (MLSys'25) | offload attention compute + KV ไป CPU สำหรับ online serving | เป้า multi-tenant throughput; เรา single user บนเดสก์ท็อป |
| KTransformers | วาง expert บน CPU พร้อม kernel AVX512/AMX เฉพาะทาง | CPU เครื่องนี้ไม่มี AVX-512/AMX ⇒ ไม่ได้กำไร |
| llama-swap / SmarterRouter / [vram_llm_orchestrator](https://github.com/vkorotkikh/vram_llm_orchestrator) | สลับโปรเซส/route ตาม VRAM ว่าง และ balance layer ตาม free VRAM | เป็น proxy/ตัวสลับ ไม่ได้แก้ปัญหา "รู้ VRAM ผิด" — และ vram_llm_orchestrator ใช้ free VRAM เหมือนกัน |

**บทเรียนร่วมของงานวิจัยกลุ่มนี้:** กำไรจริงมาจาก (ก) cache expert ที่ใช้ซ้ำ และ (ข) ไม่ย้ายข้อมูลเกินจำเป็น — ไม่ได้มาจากการเขียน planner ใหม่

---

## 4. ฝั่ง Windows/WDDM — จุดที่ทั้งวงการยังไม่แตะ

### 4.1 กลไกที่ทำให้ตัวเลขของเราเพี้ยน

- WDDM ทำ oversubscription: แอปจอง VRAM เกินได้ แล้ว OS สลับหน้าลง RAM เงียบ ๆ ⇒ `cudaMemGetInfo` เห็น "ว่าง" ไม่ตรงกับที่จะได้จริง
- `IDXGIAdapter3::QueryVideoMemoryInfo` ให้ค่าที่ถูกต้อง: `Budget` = "OS-provided video memory budget ... that the application should target" และ "If CurrentUsage is greater than Budget, the application may incur stuttering or performance penalties" — [Microsoft Learn](https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_4/ns-dxgi1_4-dxgi_query_video_memory_info)
- ไดรเวอร์ NVIDIA มีสวิตช์ปิดการ spill: **CUDA - Sysmem Fallback Policy → Prefer No Sysmem Fallback** ทำให้ allocation ล้มแทนที่จะไหลลง RAM — [runaihome](https://runaihome.com/blog/shared-gpu-memory-slow-local-ai-sysmem-fallback-fix-2026/), [NVIDIA forum](https://forums.developer.nvidia.com/t/cudamalloc-with-sysmem-fallback/347791)
- ฝั่ง Linux ต้องมี shim เลียนแบบ WDDM ถึงจะได้พฤติกรรมนี้ — [cuda-vmm-sysmem-fallback](https://github.com/AmeerJ97/cuda-vmm-sysmem-fallback)

### 4.2 วัดในเครื่องนี้ (13 พ.ย. 2026, `pwsh -File overspec/probe_budget.ps1`)

```
## GPU Process Memory (Dedicated, MB)
     790 MB  pid_2332_...
     239 MB  pid_15864_...
  --- รวมทุกโปรเซส: 1068 MB
## nvidia-smi
  total, used, reserved, free (MiB) = 8151, 644, 241, 7267
## sysmem fallback policy: ไม่พบคีย์ (ใช้ค่า default ของไดรเวอร์)
```

**ผลการลอง DXGI ผ่าน ctypes (13 พ.ย. 2026):** `QueryVideoMemoryInfo` คืน `Budget = 0` ทั้งที่สร้าง D3D12 device และ D3D11 device บน adapter แล้ว (เลือกการ์ดถูก: RTX 5050, 7910 MiB) ⇒ ช่องว่าง G1 ยังจริง แต่การใช้งานบนเครื่องนี้ต้องถอยไปใช้ `nvidia-smi free − reserved` เป็นเพดาน (`overspec/budget.py` ทำ fallback นี้แล้ว)

อ่านได้ 3 อย่างที่ไม่เคยมีใครในระบบนี้ใช้: **ใครถือ VRAM เท่าไร (แยก pid)**, **reserved ของไดรเวอร์**, และ **งบจริง** (DXGI — ยังไม่ได้เก็บค่าในรอบนี้ ต้องเรียกผ่าน COM)

ข้อสังเกตที่ต้องแก้ก่อนใช้: ผลรวม counter (1068 MB) ไม่เท่ากับ `used` ของ nvidia-smi (644 MiB) ⇒ ต้องนิยาม "ค่าอ้างอิง" ให้ชัดก่อนเอาไปคิดplacement

### 4.3 หลักฐานว่า `--fit` ไม่พอ บนเครื่องนี้เอง

| วัด | ค่า | ที่มา |
|---|---|---|
| llama.cpp คิดว่าใช้บนการ์ด | 4458 MiB | `runtime/diag.log:195` |
| VRAM เพิ่มจริงตอนโหลด | 266 MiB | `README.md:36-38` |
| ⇒ ถูก spill | ~4192 MiB | คำนวณจากสองบรรทัดบน |
| fit ตัดสินใจบนตัวเลขอะไร | "free device memory 2473 MiB" | `runtime/diag.log:22` |

⇒ ตัวเลขที่ fit ใช้ (2473 MiB) ไม่ได้บอกว่า 4192 MiB จะถูกสลับลง RAM — และไม่มีบรรทัดไหนใน log เตือน

---

## 5. ชุดข้อสอบคุณภาพ (ระยะ 2 ของแผนเดิม — ใช้ของที่มี ไม่ต้องเขียนเอง)

| ชุด | ใช้วัดอะไร | ลิงก์ |
|---|---|---|
| ThaiExam (SCB10X/Typhoon) | ความรู้ภาษาไทยจากข้อสอบจริง | [arXiv 2312.13951](https://arxiv.org/pdf/2312.13951) |
| ThaiLLM-benchmark (ฐาน SEACrowd) | มาตรฐานชุมชนไทย | [tonkaew131/ThaiLLM-benchmark](https://github.com/tonkaew131/ThaiLLM-benchmark) |
| BFCL v4 | tool calling / function calling | [leaderboard](https://llm-stats.com/benchmarks/bfcl) |
| lm-evaluation-harness | ตัวรัน (รองรับ GGUF ผ่าน backend) | ใช้เป็น harness ไม่ต้องเขียน runner เอง |

**ยังไม่มีใครต่อ:** คะแนนจากชุดเหล่านี้ → ผูกกับ config การวาง (เช่น "-ngl 32 vs -cmoe + moe-cache 2 GB, คุณภาพต่างกันเท่าไร") นี่คือช่องที่ §6 ข้อ G4

---

## 6. ช่องว่างที่เหลือ (สิ่งที่ OverSpec ควรทำจริง)

| # | ช่องว่าง | หลักฐานว่าไม่มีใครทำ | สิ่งที่ต้องมี |
|---|---|---|---|
| **G1** | ใช้งบ WDDM จริงเป็นเพดานการวาง | `fit` ใช้ free device memory (`diag.log:22`); DXGI Budget เป็นคนละค่าและถูกออกแบบมาให้เป็นเพดานที่แอปควรยึด | ตัวอ่าน Budget + policy ตั้งเพดาน |
| **G2** | ยืนยันหลังโหลดว่า weights resident จริง | ไม่มีเครื่องมือไหนรายงาน 4458 vs 266 MiB; log ไม่เตือน | ตัวตรวจ spill (จอง vs เพิ่มจริง) + exit code |
| **G3** | รู้ว่าแอปอื่นถือ VRAM เท่าไรก่อนตัดสินใจ | llama-swap/SmarterRouter ดูแค่ VRAM ว่าง; `probe_budget.ps1` พิสูจน์ว่าแยก pid ได้ | admission control + คำแนะนำปิด/ลดแอป |
| **G4** | ผูกคุณภาพเข้ากับการวาง | ชุดข้อสอบมีครบ แต่ไม่มีตัวเชื่อม | eval gate ก่อนเลื่อน config ขึ้น production |
| G5 | ซ้ำ: placement, MoE offload, expert cache, model swap | ทำแล้วทั้งหมด (§1) | **ห้ามเขียนใหม่** |

---

## 7. อ้างอิง

1. llama.cpp fit-params — https://github.com/ggml-org/llama.cpp/tree/master/tools/fit-params
2. llama.cpp `common/arg.cpp` (master) — https://github.com/ggml-org/llama.cpp/blob/master/common/arg.cpp
3. llama.cpp PR #29887 (MoE expert GPU cache) — https://github.com/ggml-org/llama.cpp/pull/29887
4. llama.cpp `common/common.h` (fit_params) — https://github.com/ggml-org/llama.cpp/blob/master/common/common.h
5. DXGI_QUERY_VIDEO_MEMORY_INFO — https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_4/ns-dxgi1_4-dxgi_query_video_memory_info
6. NVIDIA sysmem fallback (ชุมชน) — https://runaihome.com/blog/shared-gpu-memory-slow-local-ai-sysmem-fallback-fix-2026/
7. NVIDIA forum: cudaMalloc กับ sysmem fallback — https://forums.developer.nvidia.com/t/cudamalloc-with-sysmem-fallback/347791
8. cuda-vmm-sysmem-fallback (shim ฝั่ง Linux) — https://github.com/AmeerJ97/cuda-vmm-sysmem-fallback
9. Fiddler (ICLR'25) — https://github.com/efeslab/fiddler
10. MoBiLE — https://arxiv.org/abs/2510.12357
11. MoE-Infinity — https://openreview.net/forum?id=BL7WMLJKZM
12. NEO (MLSys'25) — https://mlsys.org/virtual/2025/poster/3230
13. FloE (ICML'25) — https://icml.cc/virtual/2025/poster/44378
14. vram_llm_orchestrator — https://github.com/vkorotkikh/vram_llm_orchestrator
15. ThaiExam — https://arxiv.org/pdf/2312.13951
16. ThaiLLM-benchmark — https://github.com/tonkaew131/ThaiLLM-benchmark
