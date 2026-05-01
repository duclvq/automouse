# Question/Answer Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Teach the click loop to OCR the screen each cycle, fuzzy-match the question against a stored database, and click the known answer's text directly when matched — falling back to the existing random-circle click + observe-blue-line learning pass when there's no match.

**Architecture:** Pure helpers (text normalization, fuzzy match, color mask, connected-component bbox, text-box lookup, dominant color, JSON round-trips) are added to `automouse.py` and unit-tested via `tmp_path` fixtures. Apple Vision is wrapped in a single `ocr_image(PIL.Image) -> List[(text, bbox)]` helper using `pyobjc-framework-Vision`. The click loop gets a small decision branch in `run_detection_loop` that uses these pieces; the existing rectangle-clicking phase is unchanged. All new state lives under `templates/` (`blue_sample.png`, `answers.json`, plus a `blue_rgb` key in `config.json`).

**Tech Stack:** Python 3, `pyobjc-framework-Vision` (new), `cv2` (existing), `numpy`, `PIL`, `pyautogui`, `tkinter`, `pytest`.

**Spec:** `docs/superpowers/specs/2026-05-01-question-answer-memory-design.md`

---

## File Structure

| Path | Purpose |
| --- | --- |
| `automouse.py` | Modified. New constants, ~10 pure helpers, `ocr_image` wrapper, GUI button, `try_click_known_answer`, `observe_after_click`, click-loop branch. |
| `tests/test_automouse.py` | Modified. Adds unit tests for every pure helper. |
| `requirements.txt` | Modified. Adds `pyobjc-framework-Vision==12.1`. |
| `templates/answers.json` | Created at runtime. |
| `templates/blue_sample.png` | Created at runtime when user captures the blue color. |

A shared type alias `BBox = Tuple[int, int, int, int]` is added near the existing `ROI` alias in `automouse.py`. All bounding boxes throughout this feature are in **pixel coordinates with top-left origin**, format `(x, y, w, h)`.

---

### Task 1: Dependency + new constants + BBox alias

**Files:**
- Modify: `requirements.txt`
- Modify: `automouse.py`

- [ ] **Step 1: Add the dependency**

Edit `requirements.txt` and add (alphabetically just before/after the existing `pyobjc-framework-Quartz` line):

```
pyobjc-framework-Vision==12.1
```

- [ ] **Step 2: Install**

Run: `pip install -r requirements.txt`
Expected: `pyobjc-framework-Vision-12.1` installs, plus its pyobjc deps.

- [ ] **Step 3: Verify import works**

Run: `python -c "import Vision; print('Vision OK')"`
Expected: prints `Vision OK`. If this raises, stop and report; we cannot proceed without Apple Vision.

- [ ] **Step 4: Add module-level constants and the BBox alias**

In `automouse.py`, find this block:

```python
TEMPLATES_DIR = Path("templates")
CONFIG_PATH = TEMPLATES_DIR / "config.json"
CIRCLE_PATH = TEMPLATES_DIR / "circle.png"
RECTANGLES_DIR = TEMPLATES_DIR / "rectangles"
```

Replace with:

```python
TEMPLATES_DIR = Path("templates")
CONFIG_PATH = TEMPLATES_DIR / "config.json"
CIRCLE_PATH = TEMPLATES_DIR / "circle.png"
RECTANGLES_DIR = TEMPLATES_DIR / "rectangles"
ANSWERS_PATH = TEMPLATES_DIR / "answers.json"
BLUE_SAMPLE_PATH = TEMPLATES_DIR / "blue_sample.png"
```

Find the existing `ROI = Tuple[int, int, int, int] ...` line and add directly below it:

```python
BBox = Tuple[int, int, int, int]  # (x, y, w, h) in pixels, top-left origin
```

Find the existing constants block that contains `MATCH_THRESHOLD`, `MIN_DELAY`, `MAX_DELAY`, etc., and append after `BREAK_MAX_SECONDS = 10.0`:

```python
# Question/answer memory feature.
BLUE_COLOR_TOLERANCE = 25
QUESTION_MATCH_THRESHOLD = 0.9
ANSWER_MATCH_THRESHOLD = 0.9
```

- [ ] **Step 5: Verify imports/compile/tests still pass**

Run: `python -c "import automouse" && python -m py_compile automouse.py && python -m pytest tests/test_automouse.py -v`
Expected: 20 passed.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt automouse.py
git commit -m "chore: add Vision dep and Q/A memory constants"
```

---

### Task 2: `normalize_ocr_text` (TDD)

**Files:**
- Modify: `automouse.py`
- Modify: `tests/test_automouse.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_automouse.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_automouse.py -v`
Expected: 4 new tests FAIL with import error.

- [ ] **Step 3: Implement `normalize_ocr_text`**

Append to `automouse.py` after `migrate_legacy_rectangle`:

```python
def normalize_ocr_text(observations: List[Tuple[str, BBox]]) -> str:
    """Lowercase, strip, drop blanks, dedupe, sort, join with newlines.
    Order-independent so shuffled answers don't change the signature."""
    lines = sorted({t.strip().lower()
                    for t, _ in observations
                    if t.strip()})
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_automouse.py -v`
Expected: 24 passed.

- [ ] **Step 5: Commit**

```bash
git add automouse.py tests/test_automouse.py
git commit -m "feat: normalize_ocr_text helper"
```

---

### Task 3: `find_question_match` (TDD)

**Files:**
- Modify: `automouse.py`
- Modify: `tests/test_automouse.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_automouse.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_automouse.py -v`
Expected: 4 new tests FAIL with import error.

- [ ] **Step 3: Implement `find_question_match`**

Add `from difflib import SequenceMatcher` to the top of `automouse.py` import block. Then append after `normalize_ocr_text`:

```python
def find_question_match(
    question: str,
    db: List[Dict[str, str]],
    threshold: float = QUESTION_MATCH_THRESHOLD,
) -> Optional[Dict[str, str]]:
    """Return the DB entry with highest SequenceMatcher.ratio(question,
    entry['question']) >= threshold, or None if no entry qualifies."""
    best = None
    best_ratio = threshold
    for entry in db:
        ratio = SequenceMatcher(None, question, entry["question"]).ratio()
        if ratio >= best_ratio:
            best = entry
            best_ratio = ratio
    return best
```

Add `Dict` to the existing `from typing import ...` line.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_automouse.py -v`
Expected: 28 passed.

- [ ] **Step 5: Commit**

```bash
git add automouse.py tests/test_automouse.py
git commit -m "feat: find_question_match fuzzy matcher"
```

---

### Task 4: `color_mask` (TDD)

**Files:**
- Modify: `automouse.py`
- Modify: `tests/test_automouse.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_automouse.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_automouse.py -v`
Expected: 4 new tests FAIL with import error.

- [ ] **Step 3: Implement `color_mask`**

Append to `automouse.py` after `find_question_match`:

```python
def color_mask(image: Image.Image, target_rgb: Tuple[int, int, int],
               tolerance: int = BLUE_COLOR_TOLERANCE) -> np.ndarray:
    """Return an HxW bool mask of pixels within `tolerance` per channel
    of target_rgb."""
    arr = np.asarray(image.convert("RGB"))
    diff = np.abs(arr.astype(int) - np.asarray(target_rgb, dtype=int))
    return np.all(diff <= tolerance, axis=-1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_automouse.py -v`
Expected: 32 passed.

- [ ] **Step 5: Commit**

```bash
git add automouse.py tests/test_automouse.py
git commit -m "feat: color_mask helper"
```

---

### Task 5: `largest_connected_region` (TDD)

**Files:**
- Modify: `automouse.py`
- Modify: `tests/test_automouse.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_automouse.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_automouse.py -v`
Expected: 4 new tests FAIL with import error.

- [ ] **Step 3: Implement `largest_connected_region`**

Append to `automouse.py` after `color_mask`:

```python
def largest_connected_region(mask: np.ndarray) -> Optional[BBox]:
    """Return (x, y, w, h) bbox of the largest True-region in `mask`,
    or None if no True pixels. Uses cv2.connectedComponentsWithStats."""
    if not mask.any():
        return None
    n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    # Label 0 is background. Pick the largest of labels 1..n-1.
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = int(np.argmax(areas)) + 1  # +1 because we skipped label 0
    x = int(stats[idx, cv2.CC_STAT_LEFT])
    y = int(stats[idx, cv2.CC_STAT_TOP])
    w = int(stats[idx, cv2.CC_STAT_WIDTH])
    h = int(stats[idx, cv2.CC_STAT_HEIGHT])
    return (x, y, w, h)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_automouse.py -v`
Expected: 36 passed.

- [ ] **Step 5: Commit**

```bash
git add automouse.py tests/test_automouse.py
git commit -m "feat: largest_connected_region helper"
```

---

### Task 6: `find_text_box` (TDD)

**Files:**
- Modify: `automouse.py`
- Modify: `tests/test_automouse.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_automouse.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_automouse.py -v`
Expected: 5 new tests FAIL with import error.

- [ ] **Step 3: Implement `find_text_box`**

Append to `automouse.py` after `largest_connected_region`:

```python
def find_text_box(observations: List[Tuple[str, BBox]],
                  needle: str) -> Optional[BBox]:
    """Return the bbox of the observation whose text best matches needle.
    Case-insensitive trimmed equality first; then SequenceMatcher >=
    ANSWER_MATCH_THRESHOLD. Returns None if no qualifying observation."""
    needle_norm = needle.strip().lower()
    if not needle_norm:
        return None
    best = None
    best_ratio = ANSWER_MATCH_THRESHOLD
    for text, box in observations:
        text_norm = text.strip().lower()
        if text_norm == needle_norm:
            return box
        ratio = SequenceMatcher(None, text_norm, needle_norm).ratio()
        if ratio >= best_ratio:
            best = box
            best_ratio = ratio
    return best
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_automouse.py -v`
Expected: 41 passed.

- [ ] **Step 5: Commit**

```bash
git add automouse.py tests/test_automouse.py
git commit -m "feat: find_text_box helper"
```

---

### Task 7: `text_in_region` (TDD)

**Files:**
- Modify: `automouse.py`
- Modify: `tests/test_automouse.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_automouse.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_automouse.py -v`
Expected: 4 new tests FAIL with import error.

- [ ] **Step 3: Implement `text_in_region`**

Append to `automouse.py` after `find_text_box`:

```python
def text_in_region(observations: List[Tuple[str, BBox]],
                   region: BBox) -> str:
    """Return concatenated text of observations whose bbox vertically
    overlaps `region` (Y ranges intersect by >=1 px), sorted by X."""
    rx, ry, rw, rh = region
    r_y0, r_y1 = ry, ry + rh
    overlapping: List[Tuple[int, str]] = []
    for text, (x, y, w, h) in observations:
        o_y0, o_y1 = y, y + h
        if o_y1 > r_y0 and o_y0 < r_y1:  # any vertical overlap
            overlapping.append((x, text))
    overlapping.sort(key=lambda p: p[0])
    return " ".join(t for _, t in overlapping)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_automouse.py -v`
Expected: 45 passed.

- [ ] **Step 5: Commit**

```bash
git add automouse.py tests/test_automouse.py
git commit -m "feat: text_in_region helper"
```

---

### Task 8: `dominant_color` (TDD)

**Files:**
- Modify: `automouse.py`
- Modify: `tests/test_automouse.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_automouse.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_automouse.py -v`
Expected: 2 new tests FAIL with import error.

- [ ] **Step 3: Implement `dominant_color`**

Append to `automouse.py` after `text_in_region`:

```python
def dominant_color(image: Image.Image) -> Tuple[int, int, int]:
    """Return the median (R, G, B) of all pixels."""
    arr = np.asarray(image.convert("RGB"))
    flat = arr.reshape(-1, 3)
    r, g, b = np.median(flat, axis=0).astype(int).tolist()
    return (int(r), int(g), int(b))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_automouse.py -v`
Expected: 47 passed.

- [ ] **Step 5: Commit**

```bash
git add automouse.py tests/test_automouse.py
git commit -m "feat: dominant_color helper"
```

---

### Task 9: Persistence helpers (TDD)

**Files:**
- Modify: `automouse.py`
- Modify: `tests/test_automouse.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_automouse.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_automouse.py -v`
Expected: 8 new tests FAIL with import error.

- [ ] **Step 3: Implement the four helpers**

Append to `automouse.py` after `dominant_color`:

```python
def save_blue_rgb(config_path: Path, rgb: Tuple[int, int, int]) -> None:
    """Set "blue_rgb" in config.json, preserving other keys. Creates
    parent dirs and the file if missing."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        data = json.loads(config_path.read_text())
    else:
        data = {}
    data["blue_rgb"] = [int(rgb[0]), int(rgb[1]), int(rgb[2])]
    tmp = config_path.with_suffix(config_path.suffix + ".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(config_path)


def load_blue_rgb(config_path: Path) -> Optional[Tuple[int, int, int]]:
    """Return the saved blue_rgb tuple, or None if missing/absent key."""
    if not config_path.exists():
        return None
    data = json.loads(config_path.read_text())
    rgb = data.get("blue_rgb")
    if rgb is None:
        return None
    return (int(rgb[0]), int(rgb[1]), int(rgb[2]))


def save_answers_db(path: Path, db: List[Dict[str, str]]) -> None:
    """Atomically write the answer database as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(db, ensure_ascii=False))
    tmp.replace(path)


def load_answers_db(path: Path) -> List[Dict[str, str]]:
    """Return the answer database, or [] if the file does not exist.
    Raises on JSON parse error so corrupt files surface clearly."""
    if not path.exists():
        return []
    return json.loads(path.read_text())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_automouse.py -v`
Expected: 55 passed.

- [ ] **Step 5: Commit**

```bash
git add automouse.py tests/test_automouse.py
git commit -m "feat: blue_rgb and answers.json persistence"
```

---

### Task 10: `ocr_image` Apple Vision wrapper

**Files:**
- Modify: `automouse.py`

This task does not have unit tests — Apple Vision needs a real macOS process and image. We sanity-check at the end with a synthetic image whose text we know.

- [ ] **Step 1: Implement the wrapper**

Add to the import block at the top of `automouse.py`:

```python
import Vision
import objc
from CoreGraphics import CGImageSourceCreateImageAtIndex, CGImageSourceCreateWithData
from Foundation import NSData
```

If the four-line import causes confusion later (CoreGraphics/Foundation often ship under different names depending on pyobjc version), the equivalent `from Quartz import ...` works because Quartz re-exports both `CGImageSourceCreateImageAtIndex` and `NSData`. If `from CoreGraphics import ...` fails on this machine, replace those imports with:

```python
from Quartz import CGImageSourceCreateImageAtIndex, CGImageSourceCreateWithData
from Foundation import NSData
```

Append to `automouse.py` after `load_answers_db`:

```python
def _pil_to_cgimage(image: Image.Image):
    """PIL.Image → CGImage via PNG bytes."""
    import io
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    data = NSData.dataWithBytes_length_(buf.getvalue(), len(buf.getvalue()))
    src = CGImageSourceCreateWithData(data, None)
    if src is None:
        raise RuntimeError("CGImageSourceCreateWithData returned None")
    cg = CGImageSourceCreateImageAtIndex(src, 0, None)
    if cg is None:
        raise RuntimeError("CGImageSourceCreateImageAtIndex returned None")
    return cg


def ocr_image(image: Image.Image) -> List[Tuple[str, BBox]]:
    """Run Apple Vision text recognition on a PIL image. Returns
    [(text, (x, y, w, h)), ...] in pixel coords with top-left origin.
    Returns [] when no text is found."""
    cg = _pil_to_cgimage(image)
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(True)

    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
        cg, {})
    success, error = handler.performRequests_error_([request], None)
    if not success:
        raise RuntimeError(f"Vision OCR failed: {error}")

    results = request.results() or []
    img_w, img_h = image.size
    out: List[Tuple[str, BBox]] = []
    for obs in results:
        cands = obs.topCandidates_(1)
        if not cands:
            continue
        text = str(cands[0].string())
        # boundingBox is normalized [0..1] with origin bottom-left.
        bb = obs.boundingBox()
        nx, ny, nw, nh = bb.origin.x, bb.origin.y, bb.size.width, bb.size.height
        x = int(round(nx * img_w))
        w = int(round(nw * img_w))
        h = int(round(nh * img_h))
        # Convert origin from bottom-left to top-left.
        y_top = int(round((1.0 - ny - nh) * img_h))
        out.append((text, (x, y_top, w, h)))
    return out
```

- [ ] **Step 2: Sanity-check the wrapper with a synthetic image**

Create a temporary script `_ocr_smoke.py` (do NOT commit this):

```python
from PIL import Image, ImageDraw, ImageFont
from automouse import ocr_image

img = Image.new("RGB", (400, 100), "white")
d = ImageDraw.Draw(img)
# Use the system default font; size depends on Pillow version.
d.text((10, 30), "Hello world", fill="black")
print(ocr_image(img))
```

Run: `python _ocr_smoke.py`
Expected: prints something like `[('Hello world', (X, Y, W, H))]`. Some pixel jitter is OK; the recognized string must contain "Hello world".
Then delete: `rm _ocr_smoke.py`

If the sanity check fails (returns `[]` or wrong text), stop and report. Common cause: the import block uses the wrong CoreGraphics/Quartz module name. Try the alternate form documented in Step 1.

- [ ] **Step 3: Run automated tests (regression check)**

Run: `python -m pytest tests/test_automouse.py -v`
Expected: 55 passed (no new tests added; smoke test was manual).

- [ ] **Step 4: Sanity-check imports/compile**

Run: `python -c "import automouse" && python -m py_compile automouse.py`
Expected: no output, exit 0.

- [ ] **Step 5: Commit**

```bash
git add automouse.py
git commit -m "feat: ocr_image wrapper around Apple Vision"
```

---

### Task 11: GUI — Capture-blue button + status row

**Files:**
- Modify: `automouse.py` (extend the `App` class)

- [ ] **Step 1: Add the capture-blue button**

In `App.__init__`, find:

```python
        tk.Button(root, text="Capture circle template", width=32,
                  command=self.on_capture_circle).pack(pady=4)
```

Add directly after it:

```python
        tk.Button(root, text="Capture blue line sample", width=32,
                  command=self.on_capture_blue).pack(pady=4)
```

- [ ] **Step 2: Add the `on_capture_blue` method**

Add this method to `App` (anywhere with the other `on_*` methods):

```python
    def on_capture_blue(self) -> None:
        result = _capture_rectangle(self.root)
        if result is None:
            return
        screenshot, (x, y, w, h) = result
        crop = screenshot.crop((x, y, x + w, y + h))
        BLUE_SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        crop.save(BLUE_SAMPLE_PATH)
        save_blue_rgb(CONFIG_PATH, dominant_color(crop))
        self.refresh_status()
```

- [ ] **Step 3: Update `refresh_status` to show blue color state**

Replace the existing `refresh_status` body in `App`:

```python
    def refresh_status(self) -> None:
        rect_paths = list_rectangle_templates(RECTANGLES_DIR)

        self.rect_list.delete(0, tk.END)
        for p in rect_paths:
            self.rect_list.insert(tk.END, p.name)

        def mark(p: Path) -> str:
            return "OK" if p.exists() else "missing"
        blue = load_blue_rgb(CONFIG_PATH)
        blue_str = (f"OK ({blue[0]}, {blue[1]}, {blue[2]})"
                    if blue is not None else "not set")

        self.status.config(text=(
            f"ROI:        {mark(CONFIG_PATH)}\n"
            f"Circle:     {mark(CIRCLE_PATH)}\n"
            f"Rectangles: {len(rect_paths)}\n"
            f"Blue color: {blue_str}"
        ))
        all_ready = (CONFIG_PATH.exists()
                     and CIRCLE_PATH.exists()
                     and len(rect_paths) >= 1)
        self.run_btn.config(state=tk.NORMAL if all_ready else tk.DISABLED)
```

- [ ] **Step 4: Run automated tests + import check**

Run: `python -m pytest tests/test_automouse.py -v`
Expected: 55 passed.

Run: `python -c "import automouse" && python -m py_compile automouse.py`
Expected: no output, exit 0.

- [ ] **Step 5: Commit**

```bash
git add automouse.py
git commit -m "feat: GUI button to capture blue line sample"
```

---

### Task 12: Click-loop integration

**Files:**
- Modify: `automouse.py` (add two helpers; modify `run_detection_loop`)

- [ ] **Step 1: Add `try_click_known_answer` and `observe_after_click`**

Append to `automouse.py` after `ocr_image`:

```python
def try_click_known_answer(
    shot: Image.Image,
    ocr_obs: List[Tuple[str, BBox]],
    answers_db: List[Dict[str, str]],
    roi: ROI,
    stopper: Optional["_HoldToStop"] = None,
) -> bool:
    """If OCR'd question matches a stored entry and the answer text is
    visible, click its center and return True. Else return False."""
    if not answers_db:
        return False
    question = normalize_ocr_text(ocr_obs)
    if not question:
        return False
    match = find_question_match(question, answers_db, QUESTION_MATCH_THRESHOLD)
    if match is None:
        return False
    ans_box = find_text_box(ocr_obs, match["answer"])
    if ans_box is None:
        print(f"  [memory] question matched but answer "
              f"{match['answer']!r} not visible; falling back")
        return False
    rx, ry, _, _ = roi
    x, y, w, h = ans_box
    cx = rx + x + w // 2
    cy = ry + y + h // 2
    print(f"  [memory] clicking known answer {match['answer']!r}")
    _human_move_to(cx, cy)
    pyautogui.click()
    delay = random.uniform(MIN_DELAY, MAX_DELAY)
    if stopper is not None:
        _sleep_with_check(delay, stopper)
    else:
        time.sleep(delay)
    return True


def observe_after_click(
    roi: ROI,
    blue_rgb: Tuple[int, int, int],
    question: str,
    answers_db: List[Dict[str, str]],
) -> None:
    """Re-screenshot, find the blue indicator region, OCR the row,
    append (question, answer) to the in-memory db and write to disk."""
    if not question:
        return
    shot = pyautogui.screenshot(region=roi)
    mask = color_mask(shot, blue_rgb, BLUE_COLOR_TOLERANCE)
    region = largest_connected_region(mask)
    if region is None:
        print("  [memory] no blue region found; skipping store")
        return
    obs = ocr_image(shot)
    answer = text_in_region(obs, region).strip()
    if not answer:
        print("  [memory] blue region had no overlapping text; "
              "skipping store")
        return
    for entry in answers_db:
        if entry["question"] == question and entry["answer"] == answer:
            return  # already stored
    answers_db.append({"question": question, "answer": answer})
    save_answers_db(ANSWERS_PATH, answers_db)
    print(f"  [memory] stored answer {answer!r} for question")
```

- [ ] **Step 2: Wire into `run_detection_loop`**

In `run_detection_loop`, find the block that currently does:

```python
            circle_matches = find_matches(haystack, circle_tpl, MATCH_THRESHOLD)
            picked_circle = (
                [random.choice(circle_matches)] if circle_matches else [])
            print(f"  circles:    {len(circle_matches)} match(es), "
                  f"clicking {len(picked_circle)}")
            _click_all(picked_circle, circle_tpl.shape, roi, stopper)
            if stopper.check():
                break
```

Replace it with:

```python
            circle_matches = find_matches(haystack, circle_tpl, MATCH_THRESHOLD)
            picked_circle = (
                [random.choice(circle_matches)] if circle_matches else [])

            answer_clicked = False
            ocr_obs: List[Tuple[str, BBox]] = []
            question = ""
            if blue_rgb is not None:
                ocr_obs = ocr_image(shot)
                question = normalize_ocr_text(ocr_obs)
                answer_clicked = try_click_known_answer(
                    shot, ocr_obs, answers_db, roi, stopper)

            if not answer_clicked:
                print(f"  circles:    {len(circle_matches)} match(es), "
                      f"clicking {len(picked_circle)}")
                _click_all(picked_circle, circle_tpl.shape, roi, stopper)
                if blue_rgb is not None and question:
                    observe_after_click(roi, blue_rgb, question, answers_db)

            if stopper.check():
                break
```

Also in `run_detection_loop`, find the existing local `circle_tpl = cv2.imread(...)` line. Directly **before** the existing `for tpl, name in [(circle_tpl, "circle")] + ...` validation block, add:

```python
    blue_rgb = load_blue_rgb(CONFIG_PATH)
    answers_db = load_answers_db(ANSWERS_PATH)
    if blue_rgb is None:
        print("  [memory] blue_rgb not set; question/answer feature off")
```

- [ ] **Step 3: Run automated tests + import check**

Run: `python -m pytest tests/test_automouse.py -v`
Expected: 55 passed.

Run: `python -c "import automouse" && python -m py_compile automouse.py`
Expected: no output, exit 0.

- [ ] **Step 4: Commit**

```bash
git add automouse.py
git commit -m "feat: integrate Q/A memory into click loop"
```

---

### Task 13: Manual smoke test

**Files:** none modified.

Run `python automouse.py`. Verify the new flow.

- [ ] **Step 1: Capture blue line sample**

In the GUI, click **Capture blue line sample** and drag a tight rectangle over a known blue line on screen. After release:
- Status shows `Blue color: OK (R, G, B)` with reasonable values for blue.
- `templates/blue_sample.png` exists.
- `templates/config.json` contains `"blue_rgb": [...]` alongside the
  existing `roi`.

- [ ] **Step 2: Run with feature enabled**

Click **Run**. Terminal should print:

- `Running detection loop. ROI=...`
- For each cycle, either:
  - `[memory] clicking known answer '<text>'` (when matched), OR
  - `circles: N match(es), clicking 1` followed by either
    `[memory] stored answer '<text>' for question` or
    `[memory] no blue region found; skipping store` or
    `[memory] blue region had no overlapping text; skipping store`.
- Per-rectangle match lines, as before.

After several cycles, `templates/answers.json` should contain a non-empty array of `{question, answer}` entries.

- [ ] **Step 3: Verify memory persists**

Stop the loop (hold `s` 2s). Re-run with `python automouse.py run`. The bot should immediately use stored answers when it sees those questions again.

- [ ] **Step 4: Verify feature can be disabled**

Edit `templates/config.json` and remove the `blue_rgb` key (or rename it). Run again. Loop should print `[memory] blue_rgb not set; question/answer feature off` and behave like before — random circle click each cycle, no `[memory]` lines.

- [ ] **Step 5: Verify legacy ROI/circle/rectangles still work**

Confirm the existing GUI buttons (Set ROI, Capture circle, Add rectangle, Delete selected) still function and that the Run button still gates on ROI + circle + ≥1 rectangle (not on blue color).

If all five steps pass, the feature is verified.

---

## Self-Review Notes

- **Spec coverage:**
  - `pyobjc-framework-Vision` dep + new constants/paths → Task 1.
  - `normalize_ocr_text` → Task 2.
  - `find_question_match` (≥0.9 threshold, sorted/normalized strings) → Task 3.
  - `color_mask` (RGB ±tolerance) → Task 4.
  - `largest_connected_region` (cv2 connected components) → Task 5.
  - `find_text_box` (case-insensitive, fuzzy fallback) → Task 6.
  - `text_in_region` (vertical overlap, sorted by X) → Task 7.
  - `dominant_color` (median per channel) → Task 8.
  - `save_blue_rgb` preserving other config keys + `load_blue_rgb` +
    `save/load_answers_db` (atomic write) → Task 9.
  - `ocr_image` Apple Vision wrapper → Task 10.
  - GUI capture-blue button + status row update → Task 11.
  - `try_click_known_answer`, `observe_after_click`, click-loop branch
    → Task 12.
  - Manual smoke test (with feature on/off) → Task 13.

- **Placeholders:** none. Every code step is complete.

- **Type/name consistency:** `BBox` is the same `Tuple[int, int, int, int]` shape everywhere. `BBox` and `ROI` happen to have the same shape but different semantic meaning (`ROI` is screen-absolute; `BBox` is image-local) — this is OK and matches the spec. All function signatures match across tasks: `ocr_image -> List[Tuple[str, BBox]]`, `normalize_ocr_text` accepts that exact type, `find_text_box` accepts the same, `text_in_region` accepts the same. `answers_db` is a `List[Dict[str, str]]` everywhere. `blue_rgb` is `Optional[Tuple[int, int, int]]` everywhere.
