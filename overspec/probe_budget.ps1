# probe_budget.ps1 — อ่านงบ VRAM จริงของ Windows (WDDM)
# ใช้: pwsh -File overspec/probe_budget.ps1 [-Json] [-MinMB 100]
#
# ทำไมต้องมี: llama.cpp --fit ใช้ "free device memory" จาก backend ซึ่งบน WDDM
# ไม่เท่ากับงบที่ OS ให้จริง และไม่บอกว่าแอปอื่นถือ VRAM อยู่เท่าไร
param(
    [switch]$Json,
    [int]$MinMB = 0
)

$ErrorActionPreference = 'Stop'
$counterPath = '\GPU Process Memory(*)\Dedicated Usage'
$MB = 1MB

# --- 1. VRAM แยกตามโปรเซส -------------------------------------------------
$counters = @()
$counterError = $null
try {
    $samples = (Get-Counter -Counter $counterPath -ErrorAction Stop).CounterSamples
    foreach ($s in $samples) {
        $mib = [int][math]::Round($s.CookedValue / $MB)
        if ($mib -ge $MinMB) {
            $counters += [pscustomobject]@{ instance = $s.InstanceName; mib = $mib }
        }
    }
} catch {
    $counterError = $_.Exception.Message
}

# --- 2. nvidia-smi ---------------------------------------------------------
$smi = $null
try {
    $line = (& nvidia-smi --query-gpu=memory.total,memory.used,memory.reserved,memory.free `
                --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
    if ($line) {
        $p = @($line -split ',' | ForEach-Object { $_.Trim() })
        if ($p.Count -eq 4 -and ($p -join '') -match '^\d+$') {
            $smi = [pscustomobject]@{
                total_mib    = [int]$p[0]
                used_mib     = [int]$p[1]
                reserved_mib = [int]$p[2]
                free_mib     = [int]$p[3]
            }
        }
    }
} catch {
    $smi = $null
}

# --- 3. นโยบาย sysmem fallback ของไดรเวอร์ ---------------------------------
$policy = 'default'
$policyKeys = @()
$candidateKeys = @(
    'HKLM:\SYSTEM\CurrentControlSet\Services\nvlddmkm\Global\NVTweak',
    'HKLM:\SOFTWARE\NVIDIA Corporation\Global\NVTweak'
)
foreach ($key in $candidateKeys) {
    if (-not (Test-Path $key)) { continue }
    $props = Get-ItemProperty $key
    foreach ($prop in $props.PSObject.Properties) {
        if ($prop.Name -match '(?i)sysmem|fallback') {
            $policyKeys += [pscustomobject]@{ key = $key; name = $prop.Name; value = "$($prop.Value)" }
        }
    }
}
if ($policyKeys.Count -gt 0) { $policy = 'configured' }

if ($Json) {
    $out = [pscustomobject]@{
        counters        = $counters
        counter_error   = $counterError
        smi             = $smi
        sysmem_fallback = $policy
        policy_keys     = $policyKeys
    }
    $out | ConvertTo-Json -Depth 5 -Compress
    exit 0
}

Write-Output "## GPU Process Memory (Dedicated, MB)"
if ($counterError) { Write-Output "  อ่าน counter ไม่ได้: $counterError" }
$sorted = $counters | Sort-Object mib -Descending
if (-not $sorted) { Write-Output "  (ไม่มีข้อมูล)" }
foreach ($r in ($sorted | Select-Object -First 15)) {
    Write-Output ("  {0,6} MB  {1}" -f $r.mib, $r.instance)
}
Write-Output ("  --- รวมทุกโปรเซส: {0} MB" -f (($sorted | Measure-Object -Property mib -Sum).Sum))

Write-Output ""
Write-Output "## nvidia-smi"
if ($smi) {
    Write-Output ("  total, used, reserved, free (MiB) = {0}, {1}, {2}, {3}" -f `
        $smi.total_mib, $smi.used_mib, $smi.reserved_mib, $smi.free_mib)
} else {
    Write-Output "  อ่านไม่ได้"
}

Write-Output ""
Write-Output "## sysmem fallback policy: $policy"
foreach ($k in $policyKeys) { Write-Output ("  {0} = {1}" -f $k.name, $k.value) }
