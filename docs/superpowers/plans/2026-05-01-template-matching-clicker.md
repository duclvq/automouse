# Template-Matching Auto-Clicker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `automouse.py` — a Tkinter GUI to define an ROI and capture circle/rectangle templates, plus a detect-and-click loop that uses OpenCV template matching with random 0.5–1.0 s delays.

**Architecture:** One Python file with three layers — pure utility functions (config I/O, template matching, NMS) that are unit-tested; a Tkinter GUI for ROI/template capture; and a click loop that ties them together using `pyautogui`. The first two layers are pure and unit-tested; the GUI + click loop are smoke-tested manually.

**Tech Stack:** Python 3, `pyautogui`, `opencv-python`, `numpy`, `Pillow`, `tkinter` (stdlib), `pytest`.

**Spec:** `docs/superpowers/specs/2026-05-01-template-matching-clicker-design.md`

---

## File Structure

| Path | Purpose |
| --- | --- |
| `automouse.py` | New. Contains `Config`, `find_matches`, `non_max_suppression`, GUI class, click loop, `__main__`. |
| `tests/test_automouse.py` | New. Unit tests for pure functions. |
| `requirements.txt` | New. Pinned runtime deps. |
| `templates/` | Created at runtime. Holds `circle.png`, `rectangle.png`, `config.json`. |
| `random_mouse_mover.py` | Unchanged. |

---

### Task 1: Project setup

**Files:**
- Create: `requirements.txt`
- Create: `tests/__init__.py` (empty)
- Create: `.gitignore` (if missing) entry for `templates/`

- [ ] **Step 1: Create requirements.txt**

```
pyautogui==0.9.54
opencv-python==4.10.0.84
numpy==2.1.3
Pillow==11.0.0
pytest==8.3.3
```

- [ ] **Step 2: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: All four runtime libs + pytest install without error.

- [ ] **Step 3: Create empty tests package**

Run: `touch tests/__init__.py`

- [ ] **Step 4: Update .gitignore**

Append to `.gitignore` (create if missing):

```
__pycache__/
*.pyc
.pytest_cache/
templates/
```

- [ ] **Step 5: Commit**

```bash
git add requirements.txt tests/__init__.py .gitignore
git commit -m "chore: scaffold automouse project (deps, tests dir, gitignore)"
```

---

### Task 2: Config persistence (TDD)

**Files:**
- Create: `automouse.py` (initial — only the config helpers)
- Test: `tests/test_automouse.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_automouse.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_automouse.py -v`
Expected: All four FAIL with `ImportError: cannot import name 'load_roi' from 'automouse'` (or module not found).

- [ ] **Step 3: Create automouse.py with config helpers**

Create `automouse.py`:

```python
"""Template-matching auto-clicker — see docs/superpowers/specs/2026-05-01-template-matching-clicker-design.md"""

import json
from pathlib import Path
from typing import Optional, Tuple

ROI = Tuple[int, int, int, int]  # (x, y, width, height) in screen pixels


def save_roi(config_path: Path, roi: ROI) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps({"roi": list(roi)}))


def load_roi(config_path: Path) -> Optional[ROI]:
    if not config_path.exists():
        return None
    data = json.loads(config_path.read_text())
    x, y, w, h = data["roi"]
    return (x, y, w, h)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_automouse.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add automouse.py tests/test_automouse.py
git commit -m "feat: add ROI config persistence (load/save)"
```

---

### Task 3: Non-max suppression (TDD)

**Files:**
- Modify: `automouse.py` (add `non_max_suppression`)
- Modify: `tests/test_automouse.py` (add tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_automouse.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_automouse.py -v`
Expected: 3 new tests FAIL with import error.

- [ ] **Step 3: Implement non_max_suppression**

Append to `automouse.py`:

```python
import math
from typing import List


def non_max_suppression(
    points: List[Tuple[int, int]],
    scores: List[float],
    min_distance: float,
) -> List[Tuple[int, int]]:
    """Keep highest-scoring points; drop any within min_distance of a kept point."""
    order = sorted(range(len(points)), key=lambda i: scores[i], reverse=True)
    kept: List[Tuple[int, int]] = []
    for idx in order:
        px, py = points[idx]
        if all(math.hypot(px - kx, py - ky) >= min_distance for kx, ky in kept):
            kept.append((px, py))
    return kept
```

- [ ] **Step 4: Run tests to verify all pass**

Run: `pytest tests/test_automouse.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add automouse.py tests/test_automouse.py
git commit -m "feat: add non-max suppression for template matches"
```

---

### Task 4: Template matching (TDD)

**Files:**
- Modify: `automouse.py` (add `find_matches`)
- Modify: `tests/test_automouse.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_automouse.py`:

```python
import numpy as np

from automouse import find_matches


def test_find_matches_finds_template_in_synthetic_image():
    # 200x200 black background, two 30x30 white squares at (50, 50) and (140, 100).
    haystack = np.zeros((200, 200), dtype=np.uint8)
    haystack[50:80, 50:80] = 255
    haystack[100:130, 140:170] = 255
    template = np.full((30, 30), 255, dtype=np.uint8)

    matches = find_matches(haystack, template, threshold=0.9)

    # Each match is (top_left_x, top_left_y). Centers should be near both squares.
    centers = sorted(
        (x + template.shape[1] // 2, y + template.shape[0] // 2) for x, y in matches
    )
    assert centers == [(65, 65), (155, 115)]


def test_find_matches_returns_empty_when_no_match():
    haystack = np.zeros((100, 100), dtype=np.uint8)
    template = np.full((20, 20), 255, dtype=np.uint8)
    assert find_matches(haystack, template, threshold=0.95) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_automouse.py -v`
Expected: 2 new tests FAIL with import error.

- [ ] **Step 3: Implement find_matches**

Append to `automouse.py` (add `import cv2` and `import numpy as np` at top with other imports):

```python
import cv2
import numpy as np


def find_matches(
    haystack: np.ndarray,
    template: np.ndarray,
    threshold: float,
) -> List[Tuple[int, int]]:
    """Return top-left (x, y) coords of every NMS-deduplicated match >= threshold."""
    result = cv2.matchTemplate(haystack, template, cv2.TM_CCOEFF_NORMED)
    ys, xs = np.where(result >= threshold)
    if len(xs) == 0:
        return []
    points = list(zip(xs.tolist(), ys.tolist()))
    scores = [float(result[y, x]) for x, y in points]
    th, tw = template.shape[:2]
    min_distance = min(tw, th) / 2
    return non_max_suppression(points, scores, min_distance)
```

- [ ] **Step 4: Run tests to verify all pass**

Run: `pytest tests/test_automouse.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add automouse.py tests/test_automouse.py
git commit -m "feat: add OpenCV template matching with NMS"
```

---

### Task 5: GUI skeleton (main window, status, run-disabled state)

**Files:**
- Modify: `automouse.py`

No automated test — verified manually.

- [ ] **Step 1: Add module-level constants**

Insert near the top of `automouse.py` (after imports):

```python
TEMPLATES_DIR = Path("templates")
CONFIG_PATH = TEMPLATES_DIR / "config.json"
CIRCLE_PATH = TEMPLATES_DIR / "circle.png"
RECTANGLE_PATH = TEMPLATES_DIR / "rectangle.png"

MATCH_THRESHOLD = 0.8
MIN_DELAY = 0.5
MAX_DELAY = 1.0
```

- [ ] **Step 2: Implement the GUI class skeleton**

Append to `automouse.py`:

```python
import tkinter as tk
from tkinter import messagebox


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("Automouse")
        root.geometry("280x220")

        tk.Button(root, text="Set ROI", width=28,
                  command=self.on_set_roi).pack(pady=4)
        tk.Button(root, text="Capture circle template", width=28,
                  command=self.on_capture_circle).pack(pady=4)
        tk.Button(root, text="Capture rectangle template", width=28,
                  command=self.on_capture_rectangle).pack(pady=4)
        self.run_btn = tk.Button(root, text="Run", width=28,
                                 command=self.on_run)
        self.run_btn.pack(pady=4)

        self.status = tk.Label(root, text="", justify="left")
        self.status.pack(pady=8)

        self.refresh_status()

    # Stubs — filled in by later tasks.
    def on_set_roi(self) -> None:
        messagebox.showinfo("Automouse", "Set ROI not implemented yet")

    def on_capture_circle(self) -> None:
        messagebox.showinfo("Automouse", "Capture circle not implemented yet")

    def on_capture_rectangle(self) -> None:
        messagebox.showinfo("Automouse", "Capture rectangle not implemented yet")

    def on_run(self) -> None:
        messagebox.showinfo("Automouse", "Run not implemented yet")

    def refresh_status(self) -> None:
        def mark(p: Path) -> str:
            return "OK" if p.exists() else "missing"
        text = (
            f"ROI:       {mark(CONFIG_PATH)}\n"
            f"Circle:    {mark(CIRCLE_PATH)}\n"
            f"Rectangle: {mark(RECTANGLE_PATH)}"
        )
        self.status.config(text=text)
        all_ready = (CONFIG_PATH.exists()
                     and CIRCLE_PATH.exists()
                     and RECTANGLE_PATH.exists())
        self.run_btn.config(state=tk.NORMAL if all_ready else tk.DISABLED)


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke-test the GUI**

Run: `python automouse.py`
Expected:
- A 280×220 window with four buttons and a status label.
- All three statuses show "missing".
- "Run" is disabled (greyed out).
- Each capture button opens a placeholder messagebox.
- Closing the window exits the process.

- [ ] **Step 4: Commit**

```bash
git add automouse.py
git commit -m "feat: add GUI skeleton with status label and disabled Run button"
```

---

### Task 6: Capture overlay (drag a rectangle on a fullscreen screenshot)

**Files:**
- Modify: `automouse.py` (add `_capture_rectangle` helper, wire to all three capture buttons)

- [ ] **Step 1: Add the screenshot+overlay helper**

Append to `automouse.py` (add `import time` and `import pyautogui` at top if not already there):

```python
import time
import pyautogui
from PIL import Image


def _capture_rectangle(root: tk.Tk) -> Optional[Tuple[Image.Image, Tuple[int, int, int, int]]]:
    """
    Hide root, take fullscreen screenshot, show it on a borderless Toplevel,
    let the user drag a rectangle, return (full_screenshot, (x, y, w, h)) or
    None if cancelled with Escape.
    """
    root.withdraw()
    time.sleep(0.3)
    screenshot = pyautogui.screenshot()
    sw, sh = screenshot.size

    overlay = tk.Toplevel(root)
    overlay.overrideredirect(True)
    overlay.geometry(f"{sw}x{sh}+0+0")
    overlay.attributes("-topmost", True)

    canvas = tk.Canvas(overlay, width=sw, height=sh, highlightthickness=0,
                       cursor="crosshair")
    canvas.pack()

    tk_img = _pil_to_tk(screenshot)  # keep reference
    canvas.create_image(0, 0, anchor="nw", image=tk_img)
    canvas.image = tk_img  # prevent GC

    state = {"x0": 0, "y0": 0, "rect_id": None, "result": None}

    def on_press(e):
        state["x0"], state["y0"] = e.x, e.y
        state["rect_id"] = canvas.create_rectangle(
            e.x, e.y, e.x, e.y, outline="red", width=2)

    def on_drag(e):
        if state["rect_id"] is not None:
            canvas.coords(state["rect_id"], state["x0"], state["y0"], e.x, e.y)

    def on_release(e):
        x0, y0 = state["x0"], state["y0"]
        x1, y1 = e.x, e.y
        x, y = min(x0, x1), min(y0, y1)
        w, h = abs(x1 - x0), abs(y1 - y0)
        if w >= 5 and h >= 5:
            state["result"] = (x, y, w, h)
        overlay.destroy()

    def on_escape(_e):
        state["result"] = None
        overlay.destroy()

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    overlay.bind("<Escape>", on_escape)
    overlay.focus_force()
    overlay.wait_window()

    root.deiconify()
    if state["result"] is None:
        return None
    return screenshot, state["result"]


def _pil_to_tk(image: Image.Image):
    # Local import — only needed when GUI is in use, and avoids hard-coupling at module load.
    from PIL import ImageTk
    return ImageTk.PhotoImage(image)
```

- [ ] **Step 2: Wire all three capture buttons**

Replace the three stub methods on `App` with:

```python
    def on_set_roi(self) -> None:
        result = _capture_rectangle(self.root)
        if result is None:
            return
        _, rect = result
        save_roi(CONFIG_PATH, rect)
        self.refresh_status()

    def on_capture_circle(self) -> None:
        self._capture_template_to(CIRCLE_PATH)

    def on_capture_rectangle(self) -> None:
        self._capture_template_to(RECTANGLE_PATH)

    def _capture_template_to(self, path: Path) -> None:
        result = _capture_rectangle(self.root)
        if result is None:
            return
        screenshot, (x, y, w, h) = result
        path.parent.mkdir(parents=True, exist_ok=True)
        screenshot.crop((x, y, x + w, y + h)).save(path)
        self.refresh_status()
```

- [ ] **Step 3: Smoke-test capture flow**

Run: `python automouse.py`
Steps to verify:
- Click **Set ROI** — main window hides, fullscreen screenshot appears, drag a rectangle, release. Main window returns; status now shows `ROI: OK`. `templates/config.json` exists with sane coords.
- Click **Capture circle template**, drag over a small region. Status shows `Circle: OK`. `templates/circle.png` exists and is the cropped image.
- Same for **Capture rectangle template**.
- After all three, the **Run** button enables.
- Pressing **Escape** during a capture cancels without writing.

- [ ] **Step 4: Commit**

```bash
git add automouse.py
git commit -m "feat: add fullscreen capture overlay for ROI and templates"
```

---

### Task 7: Detection-and-click loop wired to Run button

**Files:**
- Modify: `automouse.py`

- [ ] **Step 1: Add the click loop**

Append to `automouse.py` (add `import random`, `import sys` at top if missing):

```python
import random
import sys


def _click_all(matches: List[Tuple[int, int]],
               template_shape: Tuple[int, int],
               roi: ROI) -> None:
    """Click each match center with a random delay between clicks."""
    th, tw = template_shape[:2]
    rx, ry, _, _ = roi
    for mx, my in matches:
        cx = rx + mx + tw // 2
        cy = ry + my + th // 2
        pyautogui.click(cx, cy)
        time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))


def run_detection_loop() -> None:
    pyautogui.FAILSAFE = True

    roi = load_roi(CONFIG_PATH)
    if roi is None:
        print("Missing ROI config; run the GUI first.")
        return
    rx, ry, rw, rh = roi

    circle_tpl = cv2.imread(str(CIRCLE_PATH), cv2.IMREAD_GRAYSCALE)
    rect_tpl = cv2.imread(str(RECTANGLE_PATH), cv2.IMREAD_GRAYSCALE)
    if circle_tpl is None or rect_tpl is None:
        print("Missing template files; run the GUI first.")
        return

    for tpl, name in ((circle_tpl, "circle"), (rect_tpl, "rectangle")):
        th, tw = tpl.shape[:2]
        if tw > rw or th > rh:
            print(f"{name} template ({tw}x{th}) is larger than ROI ({rw}x{rh}).")
            return

    print(f"Running detection loop. ROI={roi}. Ctrl+C to stop, or move "
          f"mouse to a screen corner to trigger pyautogui failsafe.")

    try:
        while True:
            shot = pyautogui.screenshot(region=roi)
            haystack = cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2GRAY)

            circle_matches = find_matches(haystack, circle_tpl, MATCH_THRESHOLD)
            print(f"  circles:    {len(circle_matches)} match(es)")
            _click_all(circle_matches, circle_tpl.shape, roi)

            rect_matches = find_matches(haystack, rect_tpl, MATCH_THRESHOLD)
            print(f"  rectangles: {len(rect_matches)} match(es)")
            _click_all(rect_matches, rect_tpl.shape, roi)

            time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
    except pyautogui.FailSafeException:
        print("Failsafe triggered. Exiting.")
        sys.exit(0)
    except KeyboardInterrupt:
        print("Stopped by user. Bye.")
```

- [ ] **Step 2: Wire the Run button**

Replace `App.on_run` with:

```python
    def on_run(self) -> None:
        # Validate up front; the button is normally disabled when files
        # are missing, but double-check in case state drifted.
        if not (CONFIG_PATH.exists() and CIRCLE_PATH.exists()
                and RECTANGLE_PATH.exists()):
            messagebox.showerror("Automouse", "ROI or templates missing.")
            return

        roi = load_roi(CONFIG_PATH)
        rx, ry, rw, rh = roi
        for path, name in ((CIRCLE_PATH, "circle"), (RECTANGLE_PATH, "rectangle")):
            tpl = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if tpl is None:
                messagebox.showerror("Automouse", f"Could not read {path}.")
                return
            th, tw = tpl.shape[:2]
            if tw > rw or th > rh:
                messagebox.showerror(
                    "Automouse",
                    f"{name} template ({tw}x{th}) is larger than ROI "
                    f"({rw}x{rh}). Re-capture a smaller template or a "
                    f"larger ROI.")
                return

        self.root.destroy()
        run_detection_loop()
```

- [ ] **Step 3: Re-run automated tests (regression check)**

Run: `pytest tests/test_automouse.py -v`
Expected: 9 passed.

- [ ] **Step 4: Smoke-test end to end**

This step requires the macOS user to grant Accessibility + Screen
Recording permissions to the terminal app running Python.

Run: `python automouse.py`

1. Set ROI to a region of the screen that contains some recognizable shapes (e.g., a few buttons in a known app).
2. Capture a small region as the "circle" template (any distinct UI element will do for a smoke test).
3. Capture another small region as the "rectangle" template.
4. Click **Run**.
5. Verify the terminal prints match counts each cycle and the mouse clicks where it should.
6. Stop with Ctrl+C in the terminal — verify it exits cleanly with `Stopped by user. Bye.`
7. Re-run; verify it loads the saved ROI and templates without re-capturing.

- [ ] **Step 5: Commit**

```bash
git add automouse.py
git commit -m "feat: wire detection-and-click loop to Run button"
```

---

## Self-Review Notes

- Spec coverage: ROI persistence (Task 2), templates on disk (Task 6), GUI with four buttons + status + disabled-Run (Task 5), capture overlay with Escape cancel (Task 6), template matching + NMS at threshold 0.8 (Tasks 3–4), click loop with 0.5–1.0 s delays (Task 7), failsafe + KeyboardInterrupt handling (Task 7), validation of template-larger-than-ROI (Task 7). All covered.
- No placeholders — every code step shows full code.
- Names consistent across tasks: `load_roi` / `save_roi`, `non_max_suppression`, `find_matches`, `_capture_rectangle`, `_click_all`, `run_detection_loop`, `MATCH_THRESHOLD`, `MIN_DELAY`, `MAX_DELAY`, `CONFIG_PATH`, `CIRCLE_PATH`, `RECTANGLE_PATH`, `TEMPLATES_DIR`.
