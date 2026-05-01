import json
from pathlib import Path

import numpy as np

from automouse import find_matches, load_roi, non_max_suppression, save_roi


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


def test_find_matches_finds_template_in_synthetic_image():
    # TM_CCOEFF_NORMED needs variance in the template; a structured cross pattern
    # gives a sharp correlation peak at the true match locations.
    template = np.zeros((30, 30), dtype=np.uint8)
    template[10:20, :] = 255  # horizontal bar
    template[:, 10:20] = 255  # vertical bar

    haystack = np.zeros((200, 200), dtype=np.uint8)
    haystack[50:80, 50:80] = template       # top-left at (x=50, y=50)
    haystack[100:130, 140:170] = template   # top-left at (x=140, y=100)

    matches = find_matches(haystack, template, threshold=0.9)

    assert sorted(matches) == [(50, 50), (140, 100)]


def test_find_matches_returns_empty_when_no_match():
    haystack = np.zeros((100, 100), dtype=np.uint8)
    template = np.full((20, 20), 255, dtype=np.uint8)
    assert find_matches(haystack, template, threshold=0.95) == []


from automouse import list_rectangle_templates


def test_list_rectangle_templates_sorted(tmp_path: Path):
    (tmp_path / "003.png").write_bytes(b"")
    (tmp_path / "001.png").write_bytes(b"")
    (tmp_path / "002.png").write_bytes(b"")
    result = list_rectangle_templates(tmp_path)
    assert [p.name for p in result] == ["001.png", "002.png", "003.png"]


def test_list_rectangle_templates_ignores_non_matching(tmp_path: Path):
    (tmp_path / "001.png").write_bytes(b"")
    (tmp_path / "foo.png").write_bytes(b"")
    (tmp_path / "1.png").write_bytes(b"")
    (tmp_path / "001.txt").write_bytes(b"")
    result = list_rectangle_templates(tmp_path)
    assert [p.name for p in result] == ["001.png"]


def test_list_rectangle_templates_empty_dir(tmp_path: Path):
    assert list_rectangle_templates(tmp_path) == []


def test_list_rectangle_templates_missing_dir(tmp_path: Path):
    assert list_rectangle_templates(tmp_path / "does_not_exist") == []


from automouse import next_rectangle_number


def test_next_rectangle_number_empty_dir(tmp_path: Path):
    assert next_rectangle_number(tmp_path) == 1


def test_next_rectangle_number_missing_dir(tmp_path: Path):
    assert next_rectangle_number(tmp_path / "missing") == 1


def test_next_rectangle_number_with_gaps(tmp_path: Path):
    (tmp_path / "001.png").write_bytes(b"")
    (tmp_path / "003.png").write_bytes(b"")
    # max + 1, NOT first gap.
    assert next_rectangle_number(tmp_path) == 4


def test_next_rectangle_number_ignores_non_matching(tmp_path: Path):
    (tmp_path / "001.png").write_bytes(b"")
    (tmp_path / "999.txt").write_bytes(b"")
    (tmp_path / "abc.png").write_bytes(b"")
    assert next_rectangle_number(tmp_path) == 2


from automouse import migrate_legacy_rectangle


def test_migrate_legacy_rectangle_moves_file(tmp_path: Path):
    (tmp_path / "rectangle.png").write_bytes(b"old-bytes")
    migrate_legacy_rectangle(tmp_path)
    assert not (tmp_path / "rectangle.png").exists()
    assert (tmp_path / "rectangles" / "001.png").read_bytes() == b"old-bytes"


def test_migrate_legacy_rectangle_noop_if_dir_exists(tmp_path: Path):
    (tmp_path / "rectangle.png").write_bytes(b"old-bytes")
    (tmp_path / "rectangles").mkdir()
    migrate_legacy_rectangle(tmp_path)
    # Legacy file untouched, nothing copied.
    assert (tmp_path / "rectangle.png").exists()
    assert list((tmp_path / "rectangles").iterdir()) == []


def test_migrate_legacy_rectangle_noop_if_legacy_missing(tmp_path: Path):
    migrate_legacy_rectangle(tmp_path)
    assert not (tmp_path / "rectangle.png").exists()
    assert not (tmp_path / "rectangles").exists()
