# probe_budget.ps1 — อ่านงบ VRAM จริงของ Windows (WDDM) แยกตามโปรเซส
# ใช้: pwsh -File overspec/probe_budget.ps1 [-MinMB 100]
# เหตุผล: llama.cpp --fit ใช้ค่าจาก cudaMemGetInfo ("free device memory") ซึ่งบน WDDM
# ไม่เท่ากับงบที่ OS ให้จริง และไม่บอกว่าแอปอื่นถือ VRAM อยู่เท่าไร
param([int]$MinMB = 100)

$ErrorActionPreference = 'Stop'
$counter = '\GPU Process Memory(*)\Dedicated Usage'

Write-Output "## GPU Process Memory (Dedicated, MB)"
try {
    $samples = (Get-Counter -Counter $counter -ErrorAction Stop).CounterSamples
    $rows = $samples |
        Where-Object { $_.CookedValue -ge ($MinMB * 1MB) } |
        Sort-Object CookedValue -Descending |
        Select-Object -First 15
    if (-not $rows) { Write-Output "  (ไม่มีโปรเซสใดถือเกิน $MinMB MB)" }
    foreach ($r in $rows) {
        $mb = [math]::Round($r.CookedValue / 1MB)
        Write-Output ("  {0,6} MB  {1}" -f $mb, $r.InstanceName)
    }
    $sum = ($samples | Measure-Object -Property CookedValue -Sum).Sum
    Write-Output ("  --- รวมทุกโปรเซส: {0} MB" -f [math]::Round($sum / 1MB))
} catch {
    Write-Output "  อ่าน counter ไม่ได้: $($_.Exception.Message)"
}

Write-Output ""
Write-Output "## nvidia-smi"
$smi = & nvidia-smi --query-gpu=memory.total,memory.used,memory.reserved,memory.free --format=csv,noheader 2>&1
Write-Output "  total, used, reserved, free (MiB) = $smi"

Write-Output ""
Write-Output "## sysmem fallback policy (driver)"
$keys = @(
    'HKLM:\SYSTEM\CurrentControlSet\Services\nvlddmkm\Global\NVTweak',
    'HKLM:\SOFTWARE\NVIDIA Corporation\Global\NVTweak'
)
foreach ($k in $keys) {
    if (Test-Path $k) {
        $props = Get-ItemProperty $k
        $hit = $props.PSObject.Properties | Where-Object { $_.Name -match '(?i)cuda.*(fallback|sysmem)|sysmem' }
        if ($hit) { foreach ($h in $hit) { Write-Output ("  {0}: {1}" -f $h.Name, $h.Value) } }
    }
}
Write-Output "  (ถ้าไม่มีคีย์ = ยังใช้ค่า default ของไดรเวอร์)"
