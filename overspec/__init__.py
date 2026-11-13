"""OverSpec — supervisor ที่ให้ llama.cpp ทำงานแทนเราทุกอย่างที่มันทำได้

โมดูล: profile (GGUF) · budget (งบ VRAM จริง) · plan (ห่อ fit-params) ·
supervise (ตรวจ spill) · eval (ประตูคุณภาพ) · registry (config ที่ผ่านแล้ว)
"""

__all__ = ["profile", "budget", "plan", "supervise", "eval", "registry"]
__version__ = "3.0.0"
