"""Guards the eval dataset itself: every gold source exists and contains the evidence phrase."""

import json
import re
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = [json.loads(line) for line in (ROOT / "eval/golden.jsonl").read_text().splitlines()]


def _text(path: Path) -> str:
    t = path.read_text(encoding="utf-8")
    if path.suffix == ".html":
        t = BeautifulSoup(t, "lxml").get_text(" ")
    return re.sub(r"\s+", " ", t).lower()


def test_golden_size_and_ids():
    assert 30 <= len(GOLDEN) <= 60
    assert len({g["id"] for g in GOLDEN}) == len(GOLDEN)
    assert any(not g["answerable"] for g in GOLDEN)


def test_evidence_is_in_a_gold_source():
    missing = []
    for g in GOLDEN:
        if not g["answerable"]:
            continue
        texts = []
        for s in g["sources"]:
            p = ROOT / "data" / s
            assert p.exists(), f"{g['id']}: missing source {s}"
            texts.append(_text(p))
        ev = re.sub(r"\s+", " ", g["evidence"]).lower()
        if not any(ev in t for t in texts):
            missing.append(g["id"])
    assert not missing, f"evidence not found for {missing}"
