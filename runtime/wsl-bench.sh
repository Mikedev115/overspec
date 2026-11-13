#!/usr/bin/env bash
# วัด 9B บน WSL (Linux) เพื่อเทียบกับ Windows — ไม่มี WDDM spill
set -u

BIN="$HOME/llama/llama-b12321/llama-server"
SRC="/mnt/d/Models/ollama/blobs/sha256-dec52a44569a2a25341c4e4d3fee25846eed4f6f0b936278e3a3c900bb99d37c"
MODEL="$HOME/models/qwen9b.gguf"
PORT=8085

mkdir -p "$HOME/models"
if [ ! -f "$MODEL" ]; then
  echo "--- copy model ไป ext4 (อ่านจาก /mnt/d จะช้า) ---"
  cp "$SRC" "$MODEL"
fi
ls -lh "$MODEL"

echo "--- VRAM ก่อนโหลด ---"
nvidia-smi --query-gpu=memory.used --format=csv,noheader

"$BIN" -m "$MODEL" -ngl 99 -c 4096 -t 12 -fa on --port $PORT --parallel 1 -lv 1 \
  > "$HOME/models/server.log" 2>&1 &
PID=$!

ok=0
for i in $(seq 1 90); do
  if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then ok=1; break; fi
  sleep 1
done
echo "health: $ok (รอ ${i}s)"

if [ "$ok" = "1" ]; then
  echo "--- VRAM หลังโหลด ---"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader
  python3 - <<'PY'
import json, urllib.request
body = {"prompt": "อธิบายสั้น ๆ ว่า quantization คืออะไร",
        "n_predict": 128, "temperature": 0}
for r in (1, 2):
    req = urllib.request.Request(
        "http://localhost:8085/completion",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=300))
    print("รอบ %d: %.2f tok/s" % (r, d["timings"]["predicted_per_second"]))
PY
else
  echo "--- 20 บรรทัดท้าย log ---"
  tail -20 "$HOME/models/server.log"
fi

kill $PID 2>/dev/null
sleep 2
echo "--- VRAM หลังปิด ---"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
