from __future__ import annotations

import pytest

from overspec.budget import Snapshot, parse_proc_counters


def _smi(total=8151, used=644, reserved=241, free=7267):
    return {"total_mib": total, "used_mib": used, "reserved_mib": reserved, "free_mib": free}


def test_parse_proc_counters_merges_duplicate_instances():
    samples = [
        {"instance": "pid_2332_luid_0x00000000_0x0001a164_phys_0", "mib": 790},
        {"instance": "pid_2332_luid_0x00000000_0x0001a164_phys_0#1", "mib": 10},
        {"instance": "pid_15864_luid_0x00000000_0x0001a164_phys_0", "mib": 239},
        {"instance": "some_other_counter", "mib": 999},
    ]
    procs = parse_proc_counters(samples)

    assert procs[0] == {"pid": 2332, "dedicated_mib": 800}
    assert procs[1] == {"pid": 15864, "dedicated_mib": 239}
    assert len(procs) == 2


def test_available_uses_smi_when_dxgi_budget_is_zero():
    snap = Snapshot(local={"budget_mib": 0, "current_usage_mib": 0,
                           "available_for_reservation_mib": 0,
                           "current_reservation_mib": 0}, smi=_smi())
    assert snap.budget_source() == "smi"
    assert snap.available_mib() == 7267 - 241


def test_available_takes_narrowest_source():
    snap = Snapshot(local={"budget_mib": 5000, "current_usage_mib": 1000,
                           "available_for_reservation_mib": 0,
                           "current_reservation_mib": 0}, smi=_smi())
    assert snap.budget_source() == "dxgi"
    assert snap.available_mib() == 4000


def test_available_raises_without_any_source():
    with pytest.raises(RuntimeError):
        Snapshot().available_mib()


def test_our_process_mib_reads_matching_pid():
    snap = Snapshot(procs=[{"pid": 7, "dedicated_mib": 100},
                           {"pid": 9, "dedicated_mib": 250}])
    assert snap.our_process_mib(9) == 250
    assert snap.our_process_mib(1234) == 0
