import json
from pathlib import Path

import pytest

from automouse import load_roi, save_roi


def test_save_roi_writes_json(tmp_path: Path):
    config = tmp_path / "config.json"
    save_roi(config, (100, 200, 300, 400))
    data = json.loads(config.read_text())
    assert data == {"roi": [100, 200, 300, 400]}


def test_load_roi_returns_tuple(tmp_path: Path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"roi": [10, 20, 30, 40]}))
    assert load_roi(config) == (10, 20, 30, 40)


def test_load_roi_missing_returns_none(tmp_path: Path):
    assert load_roi(tmp_path / "missing.json") is None


def test_save_then_load_roundtrip(tmp_path: Path):
    config = tmp_path / "config.json"
    save_roi(config, (1, 2, 3, 4))
    assert load_roi(config) == (1, 2, 3, 4)
