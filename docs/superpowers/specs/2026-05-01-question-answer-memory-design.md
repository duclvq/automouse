# Question/Answer Memory — Design

**Date:** 2026-05-01
**Status:** Approved (pending user review of this document)
**Builds on:** `2026-05-01-template-matching-clicker-design.md`,
`2026-05-01-multi-rectangle-templates-design.md`

## Goal

When the click loop sees a question on screen it has seen before, click
the previously-correct answer instead of guessing. The bot learns
answers passively: after each random circle click, it inspects the
screen for a blue-line indicator marking the correct answer and stores
`(question_text → answer_text)` to disk. On future cycles, OCR'd
question text is compared against the database; on a ≥90% match, the
bot finds the stored answer text on the current screen and clicks it.

The feature is optional — the loop must keep working when no blue
color is configured and no answer database exists.

## Scope

- **In scope:** Apple Vision OCR helper; blue-line detection by RGB
  thresholding; `(question, answer)` JSON store; question fuzzy
  matching; new GUI button to capture the blue sample; integration
  into the existing click loop.
- **Out of scope:** multiple blue colors / multi-game support; GUI to
  view or edit the answer database (manual JSON edits only); spaced
  repetition or re-learning of changed answers; OCR of non-Latin
  scripts (Vision auto-detects, but no special handling); the
  visual-programming UI (feature D); the original "C2 — find a phrase
  and click on it" feature, which is partly subsumed but not exposed
  as a standalone feature here.

## Storage layout

```
templates/
├── config.json          # existing — adds optional "blue_rgb": [r,g,b]
├── circle.png           # existing
├── rectangles/          # existing
├── blue_sample.png      # new — cropped reference for blue color
└── answers.json         # new — JSON list of {"question", "answer"}
```

`config.json` schema after this feature:

```json
{
  "roi": [x, y, w, h],
  "blue_rgb": [r, g, b]      // optional; missing → feature disabled
}
```

`answers.json` schema:

```json
[
  {"question": "<normalized question text>", "answer": "<answer text>"}
]
```

## GUI changes

Add one button below "Capture circle template":

```
[ Set ROI                       ]
[ Capture circle template       ]
[ Capture blue line sample      ]      <- new
Rectangle templates: ...
```

**Capture blue line sample** runs the existing `_capture_rectangle`
overlay. On commit:

1. Crop the captured rectangle from the screenshot.
2. Compute the median R, G, B of the pixels (`dominant_color`).
3. Save the crop to `templates/blue_sample.png`.
4. Update `templates/config.json` with `"blue_rgb": [r, g, b]`,
   preserving the existing `"roi"`.

Status panel grows by one line:

```
ROI:        OK
Circle:     OK
Rectangles: 3
Blue color: OK / not set
```

The Run button is **not** gated on the blue color — it remains enabled
when ROI + circle + ≥1 rectangle are present. If `blue_rgb` is unset,
the loop runs as before, just without question/answer logic.

## Click loop integration

The current per-cycle structure (after Task 4 of multi-rect feature):

```
shot = screenshot(roi)
haystack = grayscale(shot)
circle_matches = find_matches(haystack, circle_tpl, threshold)
picked = [random.choice(circle_matches)] if circle_matches else []
_click_all(picked, ...)
for tpl in rectangle_templates: _click_all(...)
```

Replace the **circle-clicking step only** with the question/answer
flow. Rectangle clicking is unchanged.

```python
if blue_rgb is not None:
    answer_clicked = try_click_known_answer(shot, ocr_obs, answers_db, roi, stopper)
    if not answer_clicked:
        # Unknown question or stored answer not visible — click circle
        # and observe afterwards.
        _click_all(picked_circle, circle_tpl.shape, roi, stopper)
        observe_after_click(roi, blue_rgb, question_text, answers_db)
else:
    _click_all(picked_circle, circle_tpl.shape, roi, stopper)
```

`try_click_known_answer(shot, ocr_obs, answers_db, roi, stopper)`:

1. `question = normalize_ocr_text(ocr_obs)` — see Question normalization.
2. `match = find_question_match(question, answers_db, threshold=0.9)`.
3. If `match is None`: return `False`.
4. `ans_box = find_text_box(ocr_obs, match["answer"])`.
5. If `ans_box is None`: log "stored answer not visible"; return `False`.
6. `cx, cy = box_center(ans_box)` (in ROI pixels), then add ROI offset.
7. Click via existing `_human_move_to(cx, cy); pyautogui.click()`,
   followed by a `MIN_DELAY..MAX_DELAY` randomized sleep just like
   `_click_all` does.
8. Return `True`.

`observe_after_click(roi, blue_rgb, question_text, answers_db)`:

1. Take a fresh screenshot (`pyautogui.screenshot(region=roi)`).
2. `mask = color_mask(shot, blue_rgb, tolerance=25)`.
3. `region = largest_connected_region(mask)`. If `None`, log and return.
4. Re-OCR the fresh shot. Find OCR observations whose bounding box
   overlaps `region` vertically (Y ranges intersect). Concatenate their
   text in left-to-right order to form `answer_text`.
5. If `answer_text` is empty, log and return.
6. If `(question_text, answer_text)` is already in `answers_db`, return.
7. Append `{"question": question_text, "answer": answer_text}` and
   write `answers.json` atomically (write `.tmp`, rename).

## OCR helper (Apple Vision)

Uses `pyobjc-framework-Vision`. New module-level helper:

```python
def ocr_image(image: PIL.Image.Image) -> List[Tuple[str, Tuple[int,int,int,int]]]:
    """Return [(text, (x, y, w, h)), ...] for each VNRecognizedTextObservation,
    converting Vision's normalized origin-bottom-left bbox to pixel
    origin-top-left. text is the top candidate (`topCandidates(1)`).
    Returns [] on no observations."""
```

Implementation outline:

1. Convert PIL → bytes (PNG) → `NSData` →
   `Quartz.CGImageSourceCreateWithData` → `CGImage`.
2. Build `VNRecognizeTextRequest` with `recognitionLevel = 1` (accurate),
   `usesLanguageCorrection = True`.
3. `VNImageRequestHandler.alloc().initWithCGImage_options_(...)`,
   `performRequests_error_([request], None)`.
4. For each observation in `request.results()`, take `topCandidates_(1)[0]`,
   read `string()` and `boundingBox()` (a `CGRect` in [0,1] with
   bottom-left origin), convert to pixel `(x, y, w, h)` with top-left
   origin using image dimensions.

This wrapper is the only place pyobjc-framework-Vision is imported. If
the import fails (e.g. older macOS), wrap in try/except at the top of
`ocr_image`'s module so the rest of `automouse.py` still loads; calling
`ocr_image` raises with a clear message.

## Question normalization & matching

Both functions are pure and unit-tested.

```python
def normalize_ocr_text(observations: List[Tuple[str, BBox]]) -> str:
    """Lowercase, strip, drop blanks, sort lines, join with '\n'.
    Order-independent so shuffled answers don't change the signature."""
    lines = sorted({t.strip().lower() for t, _ in observations if t.strip()})
    return "\n".join(lines)


def find_question_match(
    question: str,
    db: List[Dict[str, str]],
    threshold: float = 0.9,
) -> Optional[Dict[str, str]]:
    """Return the best-matching DB entry whose normalized question has
    SequenceMatcher.ratio(question, entry['question']) >= threshold,
    else None. Compares with already-normalized stored question text."""
```

The DB stores **already-normalized** question strings. `observe_after_click`
normalizes before saving; `find_question_match` compares normalized to
normalized.

## Blue-line detection

Pure NumPy, no OpenCV-specific morphology:

```python
def color_mask(image: PIL.Image.Image, target_rgb: Tuple[int,int,int],
               tolerance: int = 25) -> np.ndarray:
    """Return a HxW bool mask of pixels within `tolerance` per channel
    of target_rgb."""
    arr = np.asarray(image.convert("RGB"))
    diff = np.abs(arr.astype(int) - np.asarray(target_rgb, dtype=int))
    return np.all(diff <= tolerance, axis=-1)


def largest_connected_region(mask: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
    """Return (x, y, w, h) bounding box of the largest connected
    True-region in `mask`, or None if no True pixels."""
```

`largest_connected_region` uses `scipy.ndimage.label` if scipy is
available, else falls back to `cv2.connectedComponentsWithStats`. We
already have `cv2`, so go with that to avoid a new dependency.

## Answer-text extraction

```python
def find_text_box(observations: List[Tuple[str, BBox]],
                  needle: str) -> Optional[BBox]:
    """Return the bbox of the observation whose text best matches needle
    (case-insensitive trimmed equality first; then SequenceMatcher >= 0.9
    fallback). None if no match."""


def text_in_region(observations: List[Tuple[str, BBox]],
                   region: BBox) -> str:
    """Return concatenated text of observations whose bbox vertically
    overlaps `region`, sorted by their X coordinate."""
```

"Vertically overlaps" = the Y ranges of the two boxes intersect by at
least 1 pixel.

## Constants

Add near the existing block:

```python
BLUE_COLOR_TOLERANCE = 25
QUESTION_MATCH_THRESHOLD = 0.9
ANSWER_MATCH_THRESHOLD = 0.9
ANSWERS_PATH = TEMPLATES_DIR / "answers.json"
BLUE_SAMPLE_PATH = TEMPLATES_DIR / "blue_sample.png"
```

## Atomic JSON writes

`save_answers_db(path, db)` writes to `path.with_suffix(".json.tmp")`
then `rename(path)`. Same pattern for `save_config` updates.

`load_answers_db(path)` returns `[]` if the file doesn't exist; raises
on JSON parse error (do not silently corrupt).

`save_blue_rgb(config_path, rgb)` reads existing `config.json`, sets
`blue_rgb`, writes atomically. Preserves any other keys.

## Tests

Unit tests in `tests/test_automouse.py`:

- `test_normalize_ocr_text_sorts_and_lowercases`
- `test_normalize_ocr_text_drops_blanks`
- `test_find_question_match_returns_match_above_threshold`
- `test_find_question_match_returns_none_below_threshold`
- `test_find_question_match_picks_best_above_threshold`
  (multiple candidates, the one with highest ratio is returned)
- `test_color_mask_uniform_target` — full-blue 5×5 image →
  mask all True
- `test_color_mask_within_tolerance` — pixel at target ±20 with
  tolerance 25 → mask True
- `test_color_mask_outside_tolerance` — pixel at target ±50 with
  tolerance 25 → mask False
- `test_largest_connected_region_returns_bbox` — 100×100 mask with
  one 30×10 blue stripe → returns (x, y, 30, 10)
- `test_largest_connected_region_picks_largest` — two regions, one
  larger → returns the larger
- `test_largest_connected_region_empty_mask` → None
- `test_find_text_box_exact_match`
- `test_find_text_box_fuzzy_match`
- `test_find_text_box_no_match` → None
- `test_text_in_region_collects_overlapping_left_to_right`
- `test_text_in_region_excludes_non_overlapping`
- `test_dominant_color_returns_median`
- `test_save_blue_rgb_preserves_roi` — round-trip preserving other
  config keys
- `test_save_load_answers_db_roundtrip`
- `test_load_answers_db_missing_file_returns_empty`

Apple Vision's `ocr_image` and the GUI flows are verified manually.

## Dependencies

- `pyobjc-framework-Vision==12.1` — new, added to `requirements.txt`.
- All other deps already present.

## Failure modes (recap)

| Condition | Behavior |
| --- | --- |
| `blue_rgb` not set | Feature disabled. Loop runs current behavior. |
| `answers.json` missing or `[]` | First run: every question is "new". Loop clicks circles and learns. |
| OCR returns no text | Treat as no question. Click circle; observation step also returns early. |
| Blue region not found after click | Log warning. Don't store anything. Continue to rectangles. |
| Stored answer text not visible | Log warning. Fall back to circle click + observe. |
| `pyobjc-framework-Vision` import fails | `ocr_image` raises a clear error on first use. The user is on macOS; we do not fall back to another OCR. |
| `answers.json` corrupted JSON | `load_answers_db` raises; the loop prints the error and exits. User edits the file or deletes it. |

## Backwards compatibility

- An existing `templates/config.json` with only `roi` continues to
  work. The new `blue_rgb` key is added when the user uses the new
  capture button.
- `python automouse.py run` still works. Reads `blue_rgb` and
  `answers.json` if present.
