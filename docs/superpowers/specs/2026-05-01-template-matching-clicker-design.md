# Template-Matching Auto-Clicker — Design

**Date:** 2026-05-01
**Status:** Approved (pending user review of this document)

## Goal

Add a script that detects on-screen circles and rectangles using OpenCV
template matching against user-supplied templates, then clicks every
match in a continuous loop with randomized inter-click delays.

A small Tkinter GUI lets the user define the ROI and capture the two
templates. ROI and templates persist to disk so subsequent runs reuse
them.

## Scope

- **In scope:** new file `automouse.py`; GUI for ROI + two template
  captures; detection-and-click loop; on-disk persistence under
  `templates/`.
- **Out of scope:** changes to existing `random_mouse_mover.py` (left
  untouched); arbitrary number of templates; labelled templates;
  one-shot (non-loop) mode; adjustable threshold via UI.

## File / data layout

```
automouse/
├── random_mouse_mover.py        # unchanged
├── automouse.py                 # new — GUI + detection loop
└── templates/
    ├── circle.png               # captured circle template
    ├── rectangle.png            # captured rectangle template
    └── config.json              # {"roi": [x, y, w, h]}  (screen pixels)
```

`config.json` schema:
```json
{ "roi": [x, y, width, height] }
```
Coordinates are absolute screen pixels of the top-left corner plus
width/height.

## GUI

Built with stdlib `tkinter`. Single main window with four buttons,
stacked vertically:

1. **Set ROI**
2. **Capture circle template**
3. **Capture rectangle template**
4. **Run** — disabled until `config.json`, `circle.png`, and
   `rectangle.png` all exist.

A status label below the buttons shows which artifacts exist
(`ROI: ✓  Circle: ✗  Rectangle: ✓`) and refreshes after each capture.

### Capture interaction (shared by all three capture buttons)

1. Hide the main window (`root.withdraw()`).
2. Sleep ~0.3 s so the window is fully gone before screenshot.
3. Take a fullscreen screenshot via `pyautogui.screenshot()`.
4. Open a new fullscreen, borderless `Toplevel` with the screenshot
   shown on a `Canvas` sized to the screen.
5. Mouse interaction:
   - `<ButtonPress-1>`: record start `(x0, y0)`, create a red
     rectangle outline on the canvas.
   - `<B1-Motion>`: update rectangle to current mouse position.
   - `<ButtonRelease-1>`: finalize `(x1, y1)`; normalize so
     `x0 < x1`, `y0 < y1`; close the Toplevel.
6. Persist the result:
   - **Set ROI** → write `[x0, y0, x1-x0, y1-y0]` to
     `templates/config.json`.
   - **Capture circle/rectangle template** → crop the screenshot to
     the rectangle and save as `templates/circle.png` /
     `templates/rectangle.png`.
7. Restore main window (`root.deiconify()`); refresh status label;
   re-evaluate Run button enable state.

Pressing `Escape` during capture cancels and returns to the main
window without writing anything.

### Run button

Validates templates and ROI first (sizes, files readable). If
validation fails, shows a `messagebox.showerror` and stays in the
GUI. On success, destroys the GUI window and starts the detection
loop in the same process (no threading — loop runs until Ctrl+C or
pyautogui failsafe).

## Detection + click loop

Constants near the top of `automouse.py`:

```python
MATCH_THRESHOLD = 0.8       # cv2.TM_CCOEFF_NORMED
MIN_DELAY       = 0.5       # seconds between clicks
MAX_DELAY       = 1.0
```

Loop body (one cycle):

1. Load `templates/config.json` → `roi = (x, y, w, h)`.
2. Load `circle.png` and `rectangle.png` as grayscale numpy arrays
   (cached across iterations — read once before the loop).
3. Capture the ROI: `pyautogui.screenshot(region=roi)` →
   numpy → grayscale.
4. **Click circles:** call `find_and_click(haystack, circle_tpl, roi)`.
5. **Click rectangles:** call `find_and_click(haystack, rectangle_tpl, roi)`.
6. Sleep `random.uniform(MIN_DELAY, MAX_DELAY)`; loop.

### `find_and_click(haystack, template, roi)`

1. `result = cv2.matchTemplate(haystack, template, cv2.TM_CCOEFF_NORMED)`.
2. `ys, xs = np.where(result >= MATCH_THRESHOLD)` — collect candidate
   top-left points.
3. **Non-max suppression:** sort candidates by score desc; iterate,
   keeping a point only if its center is farther than
   `min(template.w, template.h) / 2` from every already-kept point.
4. For each kept match:
   - Compute click center in screen coords:
     `(roi.x + match.x + tpl.w/2, roi.y + match.y + tpl.h/2)`.
   - `pyautogui.click(cx, cy)`.
   - `time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))`.
5. If no matches found, return immediately (no sleep — outer loop
   handles inter-cycle delay).

## Error handling

- **Run pressed with missing files:** button is disabled, but if
  somehow reached, show a `messagebox.showerror` and return.
- **Template larger than ROI:** detected by the Run button's
  validation step; shown via `messagebox` while still in the GUI.
- **`pyautogui.FailSafeException`:** caught at the top of the loop;
  print message and `sys.exit(0)`.
- **`KeyboardInterrupt`:** caught at the top of the loop; print
  message and exit cleanly.

## Dependencies

Add to a requirements list (or document in a comment at the top of
`automouse.py`):

- `pyautogui`
- `opencv-python`
- `numpy`
- `Pillow` (transitive via pyautogui, but pinned for clarity)

`tkinter` is stdlib.

## Open questions / non-decisions

None. All design decisions were resolved during brainstorming:

- ROI: GUI-defined, persisted to `config.json`.
- Templates: two fixed templates (`circle.png`, `rectangle.png`),
  saved to disk.
- Run mode: continuous loop until Ctrl+C / failsafe.
- Match threshold: fixed at 0.8 in code (constant, easy to tweak).
- New file `automouse.py`; old `random_mouse_mover.py` left alone.
