# Multi-Rectangle Templates — Design

**Date:** 2026-05-01
**Status:** Approved (pending user review of this document)
**Builds on:** `2026-05-01-template-matching-clicker-design.md`

## Goal

Support **N rectangle templates** instead of one. The user can add or
remove rectangle templates from the GUI; each cycle of the click loop
clicks all matches of every rectangle template, in the order they were
added.

The single circle template, the ROI flow, and the rest of the click
loop (1 random circle, then rectangles, then break logic) are
unchanged.

## Scope

- **In scope:** new on-disk layout under `templates/rectangles/`, GUI
  list of rectangle templates with Add/Delete, click loop iterates over
  all rectangle templates, validation per template, automatic migration
  of an existing single `templates/rectangle.png`.
- **Out of scope:** multiple circle templates; renaming/labeling
  templates; reordering after add; OCR; visual programming UI; red-line
  memory.

## Storage layout

```
templates/
├── config.json                # {"roi": [x, y, w, h]}  (unchanged)
├── circle.png                 # unchanged (single circle template)
└── rectangles/                # new
    ├── 001.png
    ├── 002.png
    └── ...
```

Filenames are zero-padded three-digit decimals. Click order is the
lexical sort of filenames (which equals numeric order because of
zero-padding).

### Allocating the next free number

When the user captures a new rectangle template, scan
`templates/rectangles/` for files matching `^\d{3}\.png$`, find the
maximum integer, and use `max + 1`. If the directory is empty or has
no matching files, use `1`. Format as `f"{n:03d}.png"`.

Gaps from deletions are fine and stay as gaps — they don't get
backfilled. The sort order remains stable; numbers grow monotonically.

### Migration from older single-rectangle layout

On GUI launch (and at the top of `run_detection_loop`), if
`templates/rectangle.png` exists and `templates/rectangles/` does not,
move `rectangle.png` → `rectangles/001.png`. Run once, idempotent.

## GUI changes

The existing main window has four buttons stacked vertically. Replace
the single "Capture rectangle template" button with a section
containing a list and Add/Delete controls:

```
[ Set ROI                       ]   ROI: OK / missing
[ Capture circle template       ]   Circle: OK / missing

Rectangle templates (N):
  ┌─────────────────────────┐
  │ 001.png                 │   <- tk.Listbox, single-select
  │ 002.png                 │
  │ 003.png                 │
  └─────────────────────────┘
[ + Add rectangle template ] [ Delete selected ]

[ Run ]
```

### Behavior

- **Add rectangle template** → invokes the existing
  `_capture_rectangle` flow. On a successful capture, saves to
  `rectangles/<next>.png` and refreshes the list.
- **Delete selected** → if a Listbox item is selected, deletes the
  corresponding file and refreshes the list. If nothing is selected,
  the button does nothing (no error popup).
- **Listbox** is populated by listing `rectangles/*.png` sorted by
  filename. Refreshed after every Add/Delete and on window load.
- **Run button** is enabled iff: `config.json` exists AND `circle.png`
  exists AND at least one file matches `rectangles/\d{3}\.png`.
- **Status label** shows three lines: `ROI: OK/missing`,
  `Circle: OK/missing`, `Rectangles: N`.

The window grows vertically to fit the listbox; suggested initial
size 320×360. Listbox shows ~6 rows with a scrollbar if more.

## Click loop changes

Replace the single-rectangle block in `run_detection_loop`:

```python
# Old
rect_matches = find_matches(haystack, rect_tpl, MATCH_THRESHOLD)
print(f"  rectangles: {len(rect_matches)} match(es)")
_click_all(rect_matches, rect_tpl.shape, roi, stopper)
```

with a loop over preloaded templates:

```python
# New
for name, tpl in rectangle_templates:
    matches = find_matches(haystack, tpl, MATCH_THRESHOLD)
    print(f"  {name}: {len(matches)} match(es)")
    _click_all(matches, tpl.shape, roi, stopper)
    if stopper.check():
        break
```

`rectangle_templates` is a list of `(filename: str, template: ndarray)`
loaded once before the main `while not stopper.check()` loop, by:

1. Listing `templates/rectangles/*.png`, sorted lexically.
2. For each path: `cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)`.
3. Skip any that fail to load with a printed warning.

If the resulting list is empty, print an error and return — the loop
needs at least one rectangle template.

## New helper functions in `automouse.py`

These are pure (no GUI, no pyautogui) and unit-testable:

| Function | Purpose |
| --- | --- |
| `list_rectangle_templates(dir: Path) -> List[Path]` | Returns sorted list of `NNN.png` paths under `dir`. |
| `next_rectangle_number(dir: Path) -> int` | Returns next free three-digit number. |
| `migrate_legacy_rectangle(templates_dir: Path) -> None` | Moves `rectangle.png` → `rectangles/001.png` if applicable. |

The Run button uses these plus `cv2.imread` and the inline
"`template.shape[1] <= roi.w and template.shape[0] <= roi.h`" check
(unchanged from the existing single-rectangle validation).

## Validation

Run button (and a parallel guard at the top of `run_detection_loop`)
must check:

1. `config.json` exists and parses → ROI tuple.
2. `circle.png` exists and `cv2.imread` succeeds.
3. At least one rectangle template in `rectangles/`.
4. Every loadable template (circle + every rectangle) fits inside the
   ROI: `template.width <= roi.w and template.height <= roi.h`.

If any check fails: `messagebox.showerror` (in GUI) or
`print` + `return` (in CLI `run` mode), naming the offending file.

## Tests

Unit tests in `tests/test_automouse.py`, all using `tmp_path`:

- `test_list_rectangle_templates_sorted` — create `003.png`, `001.png`,
  `002.png`; assert returned order is 001, 002, 003.
- `test_list_rectangle_templates_ignores_non_matching` — create
  `001.png`, `foo.png`, `1.png`; assert only `001.png` is returned.
- `test_list_rectangle_templates_empty_dir` — empty dir → `[]`.
- `test_list_rectangle_templates_missing_dir` — nonexistent dir → `[]`.
- `test_next_rectangle_number_empty` → `1`.
- `test_next_rectangle_number_with_gaps` — `001.png`, `003.png` → `4`
  (max + 1, NOT first gap).
- `test_migrate_legacy_rectangle_moves_file` — set up
  `rectangle.png`, no `rectangles/` dir; assert
  `rectangles/001.png` exists, `rectangle.png` is gone.
- `test_migrate_legacy_rectangle_noop_if_dir_exists` — both exist;
  nothing changes.
- `test_migrate_legacy_rectangle_noop_if_legacy_missing` — neither
  exists; nothing happens.

GUI flow (Add → list refresh, Delete → list refresh, Run-disabled
states) is verified manually as before.

## Dependencies

No new third-party dependencies. Uses `pathlib`, `re`, plus existing
`cv2`, `tkinter`.

## Backwards compatibility

- Existing `templates/circle.png` and `templates/config.json` keep
  working unchanged.
- Existing `templates/rectangle.png` is auto-migrated to
  `rectangles/001.png` on first launch.
- The `python automouse.py run` CLI still works.
