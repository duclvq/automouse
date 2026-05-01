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


from automouse import normalize_ocr_text


def test_normalize_ocr_text_sorts_and_lowercases():
    obs = [("Banana", (0, 0, 10, 10)),
           ("apple", (0, 0, 10, 10)),
           ("CHERRY", (0, 0, 10, 10))]
    assert normalize_ocr_text(obs) == "apple\nbanana\ncherry"


def test_normalize_ocr_text_drops_blanks_and_whitespace():
    obs = [("  apple  ", (0, 0, 10, 10)),
           ("", (0, 0, 10, 10)),
           ("   ", (0, 0, 10, 10)),
           ("banana", (0, 0, 10, 10))]
    assert normalize_ocr_text(obs) == "apple\nbanana"


def test_normalize_ocr_text_dedupes_lines():
    obs = [("apple", (0, 0, 10, 10)),
           ("APPLE", (0, 0, 10, 10)),
           ("banana", (0, 0, 10, 10))]
    assert normalize_ocr_text(obs) == "apple\nbanana"


def test_normalize_ocr_text_empty():
    assert normalize_ocr_text([]) == ""


from automouse import find_question_match


def test_find_question_match_returns_match_above_threshold():
    db = [{"question": "what is the capital of france", "answer": "paris"}]
    result = find_question_match("what is the capital of france", db, threshold=0.9)
    assert result == db[0]


def test_find_question_match_returns_none_below_threshold():
    db = [{"question": "what is the capital of france", "answer": "paris"}]
    result = find_question_match("totally unrelated text", db, threshold=0.9)
    assert result is None


def test_find_question_match_picks_best_above_threshold():
    db = [
        {"question": "the quick brown fox", "answer": "jumps"},
        {"question": "the quick brown fox jumps over", "answer": "lazy dog"},
    ]
    # Closer to the second entry.
    result = find_question_match("the quick brown fox jumps over the lazy",
                                 db, threshold=0.7)
    assert result == db[1]


def test_find_question_match_empty_db_returns_none():
    assert find_question_match("anything", [], threshold=0.9) is None


from PIL import Image as _Image
from automouse import color_mask


def test_color_mask_uniform_target():
    img = _Image.new("RGB", (5, 5), (0, 128, 255))
    mask = color_mask(img, (0, 128, 255), tolerance=0)
    assert mask.shape == (5, 5)
    assert mask.all()


def test_color_mask_within_tolerance():
    img = _Image.new("RGB", (3, 3), (10, 138, 255))  # off by 10 / 10 / 0
    mask = color_mask(img, (0, 128, 255), tolerance=25)
    assert mask.all()


def test_color_mask_outside_tolerance():
    img = _Image.new("RGB", (3, 3), (50, 200, 200))
    mask = color_mask(img, (0, 128, 255), tolerance=25)
    assert not mask.any()


def test_color_mask_mixed():
    img = _Image.new("RGB", (4, 1))
    img.putpixel((0, 0), (0, 128, 255))     # exact
    img.putpixel((1, 0), (10, 138, 245))    # within tol
    img.putpixel((2, 0), (200, 50, 0))      # outside
    img.putpixel((3, 0), (0, 128, 255))     # exact
    mask = color_mask(img, (0, 128, 255), tolerance=25)
    assert mask.tolist() == [[True, True, False, True]]


from automouse import largest_connected_region


def test_largest_connected_region_returns_bbox():
    mask = np.zeros((100, 100), dtype=bool)
    # 30 wide x 10 tall stripe at (x=20..50, y=40..50)
    mask[40:50, 20:50] = True
    bbox = largest_connected_region(mask)
    assert bbox == (20, 40, 30, 10)


def test_largest_connected_region_picks_largest():
    mask = np.zeros((100, 100), dtype=bool)
    mask[5:15, 5:25] = True       # 20 x 10 = 200 px
    mask[60:90, 60:80] = True     # 20 x 30 = 600 px
    bbox = largest_connected_region(mask)
    assert bbox == (60, 60, 20, 30)


def test_largest_connected_region_empty_mask():
    mask = np.zeros((50, 50), dtype=bool)
    assert largest_connected_region(mask) is None


def test_largest_connected_region_single_pixel():
    mask = np.zeros((10, 10), dtype=bool)
    mask[5, 7] = True
    assert largest_connected_region(mask) == (7, 5, 1, 1)


from automouse import find_text_box


def test_find_text_box_exact_match():
    obs = [("apple", (10, 10, 50, 20)),
           ("banana", (10, 40, 60, 20))]
    assert find_text_box(obs, "banana") == (10, 40, 60, 20)


def test_find_text_box_case_insensitive():
    obs = [("Banana", (5, 5, 60, 20))]
    assert find_text_box(obs, "banana") == (5, 5, 60, 20)


def test_find_text_box_strips_whitespace():
    obs = [("  banana  ", (5, 5, 60, 20))]
    assert find_text_box(obs, "banana") == (5, 5, 60, 20)


def test_find_text_box_fuzzy_match():
    obs = [("bananaa", (5, 5, 60, 20))]  # one extra char
    # SequenceMatcher.ratio("banana", "bananaa") ≈ 0.92
    assert find_text_box(obs, "banana") == (5, 5, 60, 20)


def test_find_text_box_no_match():
    obs = [("apple", (10, 10, 50, 20))]
    assert find_text_box(obs, "banana") is None


from automouse import text_in_region


def test_text_in_region_collects_overlapping_left_to_right():
    # All three observations vertically overlap region y=40..50.
    obs = [("world", (200, 40, 50, 10)),
           ("hello", (50, 40, 50, 10)),
           ("there", (130, 40, 50, 10))]
    region = (0, 40, 300, 10)
    assert text_in_region(obs, region) == "hello there world"


def test_text_in_region_excludes_non_overlapping():
    obs = [("hello", (0, 40, 50, 10)),
           ("ignore", (0, 100, 50, 10))]  # outside y range of region
    region = (0, 40, 50, 10)
    assert text_in_region(obs, region) == "hello"


def test_text_in_region_partial_vertical_overlap_kept():
    # Observation y=45..55 overlaps region y=50..60 by 5 pixels (ok).
    obs = [("hi", (0, 45, 20, 10))]
    region = (0, 50, 20, 10)
    assert text_in_region(obs, region) == "hi"


def test_text_in_region_no_observations():
    assert text_in_region([], (0, 0, 10, 10)) == ""


from automouse import dominant_color


def test_dominant_color_uniform():
    img = _Image.new("RGB", (5, 5), (10, 20, 30))
    assert dominant_color(img) == (10, 20, 30)


def test_dominant_color_returns_median_of_pixels():
    # 3 pixels at (0,0,255), 2 at (0,0,0); median per channel: (0, 0, 255)
    img = _Image.new("RGB", (5, 1))
    img.putpixel((0, 0), (0, 0, 255))
    img.putpixel((1, 0), (0, 0, 255))
    img.putpixel((2, 0), (0, 0, 255))
    img.putpixel((3, 0), (0, 0, 0))
    img.putpixel((4, 0), (0, 0, 0))
    assert dominant_color(img) == (0, 0, 255)


from automouse import (
    save_blue_rgb,
    load_blue_rgb,
    save_answers_db,
    load_answers_db,
)


def test_save_blue_rgb_preserves_other_keys(tmp_path: Path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"roi": [10, 20, 30, 40]}))
    save_blue_rgb(config, (0, 128, 255))
    data = json.loads(config.read_text())
    assert data == {"roi": [10, 20, 30, 40], "blue_rgb": [0, 128, 255]}


def test_save_blue_rgb_creates_file_if_missing(tmp_path: Path):
    config = tmp_path / "config.json"
    save_blue_rgb(config, (1, 2, 3))
    assert json.loads(config.read_text()) == {"blue_rgb": [1, 2, 3]}


def test_load_blue_rgb_returns_tuple(tmp_path: Path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(
        {"roi": [10, 20, 30, 40], "blue_rgb": [0, 128, 255]}))
    assert load_blue_rgb(config) == (0, 128, 255)


def test_load_blue_rgb_missing_key_returns_none(tmp_path: Path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"roi": [1, 2, 3, 4]}))
    assert load_blue_rgb(config) is None


def test_load_blue_rgb_missing_file_returns_none(tmp_path: Path):
    assert load_blue_rgb(tmp_path / "missing.json") is None


def test_save_load_answers_db_roundtrip(tmp_path: Path):
    db = [{"question": "q1", "answer": "a1"},
          {"question": "q2", "answer": "a2"}]
    path = tmp_path / "answers.json"
    save_answers_db(path, db)
    assert load_answers_db(path) == db


def test_load_answers_db_missing_file_returns_empty(tmp_path: Path):
    assert load_answers_db(tmp_path / "missing.json") == []


def test_save_answers_db_atomic_no_partial_file(tmp_path: Path):
    # After a successful save there should be no .tmp file lingering.
    path = tmp_path / "answers.json"
    save_answers_db(path, [{"question": "q", "answer": "a"}])
    assert not (tmp_path / "answers.json.tmp").exists()
    assert path.exists()
