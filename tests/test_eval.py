from __future__ import annotations

import pytest

from overspec import eval as ev


def test_dataset_has_25_items_covering_three_tags():
    items = ev.load_items()
    assert len(items) == 25
    tags = {i.tag for i in items}
    assert tags == {"thai", "code", "tool"}
    assert len({i.id for i in items}) == 25


def test_every_regex_in_dataset_compiles():
    for item in ev.load_items():
        if item.expect.get("type") == "regex":
            ev.score("x", item.expect)  # ต้องไม่โยน error


def test_score_kinds():
    assert ev.score("  56 ", {"type": "exact", "value": "56"})
    assert not ev.score("56", {"type": "exact", "value": " 56"})
    assert ev.score("คำตอบคือ 56", {"type": "contains", "value": "56"})
    assert ev.score("ปี 2568", {"type": "regex", "value": "25\\d\\d"})


def test_score_rejects_bad_or_long_pattern():
    with pytest.raises(ValueError):
        ev.score("x", {"type": "regex", "value": "("})
    with pytest.raises(ValueError):
        ev.score("x", {"type": "regex", "value": "a" * 501})


def test_gate_allows_two_regressions_only():
    assert ev.gate(25, 23)["passed"] is True
    assert ev.gate(25, 22)["passed"] is False
    assert ev.gate(20, 25)["delta"] == 5


def test_local_url_only_allows_loopback():
    assert ev.local_url("http://127.0.0.1:8099/") == "http://127.0.0.1:8099"
    assert ev.local_url("http://localhost:8080")
    with pytest.raises(ValueError):
        ev.local_url("http://10.0.0.5:8099")
    with pytest.raises(ValueError):
        ev.local_url("file:///etc/passwd")
