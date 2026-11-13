"""overspec.budget — งบ VRAM จริงของ Windows

อ่านสามแหล่งแล้วรวมเป็น snapshot เดียว:
  1. DXGI IDXGIAdapter3::QueryVideoMemoryInfo -> Budget / CurrentUsage / Reservation
     (ค่าที่ Microsoft นิยามว่าเป็นเพดานที่แอป "ควร" ยึด ไม่ใช่แค่ VRAM ที่ว่าง)
  2. Windows performance counter "GPU Process Memory" -> VRAM แยกตามโปรเซส
  3. nvidia-smi -> total/used/reserved/free + สถานะนโยบาย sysmem fallback

ใช้: python -m overspec.budget [--human]
"""
from __future__ import annotations

import ctypes
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

MIB = 1024 * 1024
HERE = Path(__file__).resolve().parent
PROBE_SCRIPT = HERE / "probe_budget.ps1"

PID_RE = re.compile(r"pid_(\d+)")


# --------------------------------------------------------------------------
# DXGI
# --------------------------------------------------------------------------

class DXGI_QUERY_VIDEO_MEMORY_INFO(ctypes.Structure):
    _fields_ = [
        ("Budget", ctypes.c_uint64),
        ("CurrentUsage", ctypes.c_uint64),
        ("AvailableForReservation", ctypes.c_uint64),
        ("CurrentReservation", ctypes.c_uint64),
    ]


class DXGI_ADAPTER_DESC1(ctypes.Structure):
    _fields_ = [
        ("Description", ctypes.c_wchar * 128),
        ("VendorId", ctypes.c_uint32),
        ("DeviceId", ctypes.c_uint32),
        ("SubSysId", ctypes.c_uint32),
        ("Revision", ctypes.c_uint32),
        ("DedicatedVideoMemory", ctypes.c_size_t),
        ("DedicatedSystemMemory", ctypes.c_size_t),
        ("SharedSystemMemory", ctypes.c_size_t),
        ("AdapterLuid", ctypes.c_uint64),
        ("Flags", ctypes.c_uint32),
    ]


DXGI_MEMORY_SEGMENT_GROUP_LOCAL = 0
DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL = 1

# {770aae78-f26f-4dba-a829-253c83d1b387}
IID_IDXGIFACTORY1 = bytes.fromhex("78ae0a776ff2ba4da829253c83d1b387")

# ดัชนี vtable: IUnknown(0-2) IDXGIObject(3-6) IDXGIFactory(7-11) IDXGIFactory1(12-13)
VT_ENUM_ADAPTERS1 = 12
# IUnknown(0-2) IDXGIObject(3-6) IDXGIAdapter(7-9) Adapter1(10) Adapter2(11) Adapter3(12-16)
VT_GET_DESC1 = 10
VT_QUERY_VIDEO_MEMORY_INFO = 13
VT_RELEASE = 2


def _com_method(ptr: ctypes.c_void_p, index: int, restype, *argtypes):
    vtable = ctypes.cast(ptr, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    proto = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)
    return proto(vtable[index])


D3D_DRIVER_TYPE_UNKNOWN = 0
D3D11_SDK_VERSION = 7
D3D_FEATURE_LEVEL_11_0 = 0xB000
# {189819f1-1db6-4b57-be54-1821339b85f7}
IID_ID3D12DEVICE = bytes.fromhex("f1199818b61d4b57be541821339b85f7")


def _create_d3d12_device(adapter: ctypes.c_void_p):
    """สร้าง D3D12 device — WDDM กำหนดงบหน่วยความจำให้โปรเซสที่มี device แบบนี้"""
    try:
        d3d12 = ctypes.WinDLL("d3d12")
    except OSError as exc:
        return None, f"โหลด d3d12.dll ไม่ได้: {exc}"
    create = d3d12.D3D12CreateDevice
    create.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                       ctypes.POINTER(ctypes.c_void_p)]
    create.restype = ctypes.c_long
    device = ctypes.c_void_p()
    iid = (ctypes.c_ubyte * 16).from_buffer_copy(IID_ID3D12DEVICE)
    hr = create(adapter, D3D_FEATURE_LEVEL_11_0, ctypes.byref(iid), ctypes.byref(device))
    if hr != 0 or not device:
        return None, f"D3D12CreateDevice ล้มเหลว hr=0x{hr & 0xFFFFFFFF:08x}"
    return device, None


def _create_d3d11_device(adapter: ctypes.c_void_p):
    """สร้าง D3D11 device บน adapter ที่เลือก

    DXGI จะให้ค่า Budget เป็น 0 จนกว่าโปรเซสจะมี D3D device ของตัวเอง
    คืน (device, context, error) — error เป็น None เมื่อสำเร็จ
    """
    try:
        d3d11 = ctypes.WinDLL("d3d11")
    except OSError as exc:
        return None, None, f"โหลด d3d11.dll ไม่ได้: {exc}"

    create = d3d11.D3D11CreateDevice
    create.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint,
        ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint,
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    create.restype = ctypes.c_long

    device = ctypes.c_void_p()
    context = ctypes.c_void_p()
    feature_level = ctypes.c_uint()
    hr = create(adapter, D3D_DRIVER_TYPE_UNKNOWN, None, 0, None, 0,
                D3D11_SDK_VERSION, ctypes.byref(device), ctypes.byref(feature_level),
                ctypes.byref(context))
    if hr != 0 or not device:
        return None, None, f"D3D11CreateDevice ล้มเหลว hr=0x{hr & 0xFFFFFFFF:08x}"
    return device, context, None


def _com_release(ptr: ctypes.c_void_p) -> None:
    if not ptr:
        return
    _com_method(ptr, VT_RELEASE, ctypes.c_ulong)(ptr)


def dxgi_read() -> dict:
    """คืนค่าจาก DXGI ของ adapter แรก; โยน OSError ถ้าเรียกไม่สำเร็จ"""
    if sys.platform != "win32":
        raise OSError("DXGI ใช้ได้เฉพาะ Windows")

    dxgi = ctypes.WinDLL("dxgi")
    create_factory = dxgi.CreateDXGIFactory1
    create_factory.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    create_factory.restype = ctypes.c_long

    factory = ctypes.c_void_p()
    iid = (ctypes.c_ubyte * 16).from_buffer_copy(IID_IDXGIFACTORY1)
    hr = create_factory(ctypes.byref(iid), ctypes.byref(factory))
    if hr != 0 or not factory:
        raise OSError(f"CreateDXGIFactory1 ล้มเหลว hr=0x{hr & 0xFFFFFFFF:08x}")

    adapters: list[tuple[ctypes.c_void_p, DXGI_ADAPTER_DESC1]] = []
    device = context = None
    enum_fn = _com_method(factory, VT_ENUM_ADAPTERS1, ctypes.c_long,
                          ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p))
    try:
        index = 0
        while True:
            adapter = ctypes.c_void_p()
            hr = enum_fn(factory, index, ctypes.byref(adapter))
            if hr != 0 or not adapter:
                break
            desc = DXGI_ADAPTER_DESC1()
            desc_fn = _com_method(adapter, VT_GET_DESC1, ctypes.c_long,
                                  ctypes.POINTER(DXGI_ADAPTER_DESC1))
            desc_fn(adapter, ctypes.byref(desc))
            adapters.append((adapter, desc))
            index += 1
        if not adapters:
            raise OSError("ไม่พบ DXGI adapter")

        # เลือกการ์ดที่มี VRAM เฉพาะตัวมากที่สุด (iGPU ของ Intel รายงาน 0-128 MiB)
        adapter, desc = max(adapters, key=lambda pair: pair[1].DedicatedVideoMemory)

        # budget ของ WDDM ถูกกำหนดตอนโปรเซสสร้าง D3D device — ไม่มี device = ได้ 0
        device, device_error = _create_d3d12_device(adapter)
        context = None
        if device_error:
            device, context, d3d11_error = _create_d3d11_device(adapter)
            device_error = f"D3D12: {device_error}; D3D11: {d3d11_error}" if d3d11_error else None

        query = _com_method(adapter, VT_QUERY_VIDEO_MEMORY_INFO, ctypes.c_long,
                            ctypes.c_uint32, ctypes.c_uint32,
                            ctypes.POINTER(DXGI_QUERY_VIDEO_MEMORY_INFO))
        out = {
            "adapter": desc.Description,
            "dedicated_vram_mib": desc.DedicatedVideoMemory // MIB,
        }
        if device_error:
            out["device_error"] = device_error
        for name, group in (("local", DXGI_MEMORY_SEGMENT_GROUP_LOCAL),
                            ("non_local", DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL)):
            info = DXGI_QUERY_VIDEO_MEMORY_INFO()
            hr = query(adapter, 0, group, ctypes.byref(info))
            if hr != 0:
                raise OSError(f"QueryVideoMemoryInfo({name}) ล้มเหลว hr=0x{hr & 0xFFFFFFFF:08x}")
            out[name] = {
                "budget_mib": info.Budget // MIB,
                "current_usage_mib": info.CurrentUsage // MIB,
                "available_for_reservation_mib": info.AvailableForReservation // MIB,
                "current_reservation_mib": info.CurrentReservation // MIB,
            }
        if not out["local"]["budget_mib"]:
            out["budget_error"] = "DXGI คืน Budget = 0 (device ยังไม่ถูกกำหนดงบ)"
        return out
    finally:
        _com_release(context)
        _com_release(device)
        for adapter_ptr, _ in adapters:
            _com_release(adapter_ptr)
        _com_release(factory)


# --------------------------------------------------------------------------
# Windows performance counters + nvidia-smi
# --------------------------------------------------------------------------

def probe_powershell(script: Path = PROBE_SCRIPT, timeout: int = 30) -> dict:
    """เรียก probe_budget.ps1 -Json แล้วคืน dict (คืน {} ถ้ารันไม่ได้)"""
    if not script.exists():
        return {}
    try:
        res = subprocess.run(
            ["pwsh", "-NoProfile", "-NonInteractive", "-File", str(script), "-Json"],
            capture_output=True, text=True, timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if res.returncode != 0 or not res.stdout.strip():
        return {}
    try:
        return json.loads(res.stdout)
    except json.JSONDecodeError:
        return {}


def parse_proc_counters(samples: list[dict]) -> list[dict]:
    """รวม instance ที่ pid เดียวกัน (Windows แตก instance เป็น #1, #2)"""
    per_pid: dict[int, int] = {}
    for s in samples:
        m = PID_RE.search(s.get("instance", ""))
        if not m:
            continue
        pid = int(m.group(1))
        per_pid[pid] = per_pid.get(pid, 0) + int(s.get("mib", 0))
    return [
        {"pid": pid, "dedicated_mib": mib}
        for pid, mib in sorted(per_pid.items(), key=lambda kv: -kv[1])
    ]


def nvidia_smi() -> dict | None:
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.reserved,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if res.returncode != 0:
        return None
    line = res.stdout.strip().splitlines()[0] if res.stdout.strip() else ""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 4 or not all(p.isdigit() for p in parts):
        return None
    total, used, reserved, free = (int(p) for p in parts)
    return {"total_mib": total, "used_mib": used, "reserved_mib": reserved, "free_mib": free}


# --------------------------------------------------------------------------
# snapshot
# --------------------------------------------------------------------------

@dataclass
class Snapshot:
    adapter: str | None = None
    dedicated_vram_mib: int | None = None
    local: dict | None = None
    non_local: dict | None = None
    procs: list[dict] = field(default_factory=list)
    procs_total_mib: int = 0
    smi: dict | None = None
    sysmem_fallback: str = "unknown"
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "adapter": self.adapter,
            "dedicated_vram_mib": self.dedicated_vram_mib,
            "local": self.local,
            "non_local": self.non_local,
            "procs": self.procs,
            "procs_total_mib": self.procs_total_mib,
            "smi": self.smi,
            "sysmem_fallback": self.sysmem_fallback,
            "errors": self.errors,
        }

    def our_process_mib(self, pid: int) -> int:
        for p in self.procs:
            if p["pid"] == pid:
                return int(p["dedicated_mib"])
        return 0

    def budget_source(self) -> str:
        if self.local and self.local["budget_mib"] > 0:
            return "dxgi"
        if self.smi:
            return "smi"
        return "none"

    def available_mib(self) -> int:
        """เพดานที่ควรใช้ตั้ง budget: แคบที่สุดของ DXGI budget กับ smi.free − reserved

        DXGI Budget คืน 0 ได้บนเครื่องนี้ (วัฒน์แล้ว 13 พ.ย. 2026) จึงต้องมีทางถอย
        """
        candidates = []
        if self.local and self.local["budget_mib"] > 0:
            candidates.append(self.local["budget_mib"] - self.local["current_usage_mib"])
        if self.smi:
            candidates.append(self.smi["free_mib"] - self.smi["reserved_mib"])
        if not candidates:
            raise RuntimeError("ไม่มีแหล่งข้อมูลหน่วยความจำที่ใช้ได้")
        return min(candidates)


def snapshot() -> Snapshot:
    snap = Snapshot()
    try:
        dx = dxgi_read()
        snap.adapter = dx["adapter"]
        snap.dedicated_vram_mib = dx["dedicated_vram_mib"]
        snap.local = dx["local"]
        snap.non_local = dx["non_local"]
        if dx.get("device_error"):
            snap.errors.append(f"dxgi: {dx['device_error']}")
        if dx.get("budget_error"):
            snap.errors.append("dxgi: budget = 0 — ใช้ nvidia-smi (free − reserved) เป็นเพดานแทน")
    except Exception as exc:  # DXGI ล้มไม่ควรทำให้ทั้ง snapshot ล้ม
        snap.errors.append(f"dxgi: {exc}")

    probe = probe_powershell()
    if probe:
        snap.procs = parse_proc_counters(probe.get("counters", []))
        snap.procs_total_mib = sum(p["dedicated_mib"] for p in snap.procs)
        snap.sysmem_fallback = probe.get("sysmem_fallback", "unknown")
        smi = probe.get("smi")
        if smi and all(isinstance(smi.get(k), int) for k in
                       ("total_mib", "used_mib", "reserved_mib", "free_mib")):
            snap.smi = smi
    else:
        snap.errors.append("powershell: probe_budget.ps1 ไม่คืนข้อมูล")

    if not snap.smi:
        snap.smi = nvidia_smi()
        if not snap.smi:
            snap.errors.append("nvidia-smi: อ่านไม่ได้")
    return snap


def render_human(snap: Snapshot) -> str:
    lines = []
    if snap.adapter:
        lines.append(f"adapter: {snap.adapter} ({snap.dedicated_vram_mib} MiB dedicated)")
    if snap.local:
        loc = snap.local
        lines.append(
            f"DXGI local  budget={loc['budget_mib']} MiB  usage={loc['current_usage_mib']} MiB  "
            f"reservation={loc['current_reservation_mib']} MiB  "
            f"availableForReservation={loc['available_for_reservation_mib']} MiB")
    if snap.non_local:
        nl = snap.non_local
        lines.append(f"DXGI shared budget={nl['budget_mib']} MiB  usage={nl['current_usage_mib']} MiB")
    if snap.smi:
        s = snap.smi
        lines.append(f"nvidia-smi   total={s['total_mib']} used={s['used_mib']} "
                     f"reserved={s['reserved_mib']} free={s['free_mib']} MiB")
    lines.append(f"sysmem fallback policy: {snap.sysmem_fallback}")
    lines.append(f"per-process (บนสุด 8): total={snap.procs_total_mib} MiB")
    for p in snap.procs[:8]:
        lines.append(f"  {p['dedicated_mib']:>6} MiB  pid {p['pid']}")
    try:
        lines.append(f"เพดานที่ควรใช้: {snap.available_mib()} MiB")
    except RuntimeError as exc:
        lines.append(f"เพดานที่ควรใช้: ไม่ทราบ ({exc})")
    if snap.errors:
        lines.append("ข้อผิดพลาด: " + "; ".join(snap.errors))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    snap = snapshot()
    if "--human" in argv:
        print(render_human(snap))
    else:
        print(json.dumps(snap.as_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
