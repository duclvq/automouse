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


from automouse import non_max_suppression


def test_nms_keeps_highest_score_when_overlapping():
    # Two candidates whose centers are 5px apart, min_distance = 10
    # Expect only the higher-scoring one survives.
    points = [(0, 0), (5, 0)]
    scores = [0.8, 0.95]
    kept = non_max_suppression(points, scores, min_distance=10)
    assert kept == [(5, 0)]


def test_nms_keeps_both_when_far_apart():
    points = [(0, 0), (100, 100)]
    scores = [0.8, 0.9]
    kept = non_max_suppression(points, scores, min_distance=10)
    # Order should be by score desc.
    assert set(kept) == {(0, 0), (100, 100)}
    assert kept[0] == (100, 100)


def test_nms_empty_input():
    assert non_max_suppression([], [], min_distance=10) == []
