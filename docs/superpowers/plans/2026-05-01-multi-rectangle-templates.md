# Multi-Rectangle Templates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single `templates/rectangle.png` with a directory of N rectangle templates managed via a GUI list, and have each click cycle iterate over all of them in order.

**Architecture:** Three pure helpers (`list_rectangle_templates`, `next_rectangle_number`, `migrate_legacy_rectangle`) added to `automouse.py` and unit-tested. The click loop pre-loads every template once and iterates per cycle. The GUI replaces the single "Capture rectangle template" button with a `tk.Listbox` plus Add/Delete buttons. Backwards-compatible: an existing `rectangle.png` is auto-migrated to `rectangles/001.png`.

**Tech Stack:** Python 3, `tkinter` (stdlib), `cv2`, `numpy`, `pyautogui`, `pytest`. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-01-multi-rectangle-templates-design.md`

---

## File Structure

| Path | Purpose |
| --- | --- |
| `automouse.py` | Modified. New constant `RECTANGLES_DIR`; new helpers; `run_detection_loop` iterates over the rectangles directory; GUI swaps the single rectangle button for a list. |
| `tests/test_automouse.py` | Modified. Adds unit tests for the three pure helpers. |
| `templates/rectangles/` | New directory created at runtime. |

No existing files are removed. The existing `RECTANGLE_PATH = TEMPLATES_DIR / "rectangle.png"` constant is removed because the migration helper handles the legacy path inline.

---

### Task 1: `list_rectangle_templates` helper (TDD)

**Files:**
- Modify: `automouse.py`
- Modify: `tests/test_automouse.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_automouse.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_automouse.py -v`
Expected: 4 new tests FAIL with `ImportError: cannot import name 'list_rectangle_templates' from 'automouse'`.

- [ ] **Step 3: Implement `list_rectangle_templates`**

Add `import re` to the existing import block at the top of `automouse.py`. Then append after the `find_matches` function:

```python
_RECT_NAME_RE = re.compile(r"^(\d{3})\.png$")


def list_rectangle_templates(directory: Path) -> List[Path]:
    """Return zero-padded NNN.png files under directory, sorted by name."""
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and _RECT_NAME_RE.match(p.name)
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_automouse.py -v`
Expected: 13 passed (9 existing + 4 new).

- [ ] **Step 5: Commit**

```bash
git add automouse.py tests/test_automouse.py
git commit -m "feat: list_rectangle_templates helper"
```

---

### Task 2: `next_rectangle_number` helper (TDD)

**Files:**
- Modify: `automouse.py`
- Modify: `tests/test_automouse.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_automouse.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_automouse.py -v`
Expected: 4 new tests FAIL with import error.

- [ ] **Step 3: Implement `next_rectangle_number`**

Append to `automouse.py` after `list_rectangle_templates`:

```python
def next_rectangle_number(directory: Path) -> int:
    """Return max existing NNN integer + 1, or 1 if none."""
    paths = list_rectangle_templates(directory)
    if not paths:
        return 1
    nums = [int(_RECT_NAME_RE.match(p.name).group(1)) for p in paths]
    return max(nums) + 1
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_automouse.py -v`
Expected: 17 passed.

- [ ] **Step 5: Commit**

```bash
git add automouse.py tests/test_automouse.py
git commit -m "feat: next_rectangle_number helper"
```

---

### Task 3: `migrate_legacy_rectangle` helper (TDD)

**Files:**
- Modify: `automouse.py`
- Modify: `tests/test_automouse.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_automouse.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_automouse.py -v`
Expected: 3 new tests FAIL with import error.

- [ ] **Step 3: Implement `migrate_legacy_rectangle`**

Append to `automouse.py` after `next_rectangle_number`:

```python
def migrate_legacy_rectangle(templates_dir: Path) -> None:
    """If templates_dir/rectangle.png exists and rectangles/ does not,
    move the legacy file to rectangles/001.png. Idempotent."""
    legacy = templates_dir / "rectangle.png"
    new_dir = templates_dir / "rectangles"
    if not legacy.exists() or new_dir.exists():
        return
    new_dir.mkdir(parents=True)
    legacy.rename(new_dir / "001.png")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_automouse.py -v`
Expected: 20 passed.

- [ ] **Step 5: Commit**

```bash
git add automouse.py tests/test_automouse.py
git commit -m "feat: migrate_legacy_rectangle helper"
```

---

### Task 4: Click loop iterates over the rectangles directory

**Files:**
- Modify: `automouse.py` (add `RECTANGLES_DIR` constant; rewrite `run_detection_loop`; remove `RECTANGLE_PATH`)

- [ ] **Step 1: Add `RECTANGLES_DIR` and remove `RECTANGLE_PATH`**

In `automouse.py`, find the existing constants block:

```python
TEMPLATES_DIR = Path("templates")
CONFIG_PATH = TEMPLATES_DIR / "config.json"
CIRCLE_PATH = TEMPLATES_DIR / "circle.png"
RECTANGLE_PATH = TEMPLATES_DIR / "rectangle.png"
```

Replace with:

```python
TEMPLATES_DIR = Path("templates")
CONFIG_PATH = TEMPLATES_DIR / "config.json"
CIRCLE_PATH = TEMPLATES_DIR / "circle.png"
RECTANGLES_DIR = TEMPLATES_DIR / "rectangles"
```

- [ ] **Step 2: Rewrite `run_detection_loop` to load all rectangles**

Replace the existing body of `run_detection_loop` from `pyautogui.FAILSAFE = True` down to the `pyautogui.FailSafeException` handler. The new body:

```python
def run_detection_loop() -> None:
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0  # we manage our own per-step delays

    migrate_legacy_rectangle(TEMPLATES_DIR)

    roi = load_roi(CONFIG_PATH)
    if roi is None:
        print("Missing ROI config; run the GUI first.")
        return
    rx, ry, rw, rh = roi

    circle_tpl = cv2.imread(str(CIRCLE_PATH), cv2.IMREAD_GRAYSCALE)
    if circle_tpl is None:
        print("Missing circle template; run the GUI first.")
        return

    rectangle_paths = list_rectangle_templates(RECTANGLES_DIR)
    if not rectangle_paths:
        print("No rectangle templates; run the GUI first.")
        return

    rectangle_templates: List[Tuple[str, np.ndarray]] = []
    for path in rectangle_paths:
        tpl = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if tpl is None:
            print(f"Could not read {path}, skipping.")
            continue
        rectangle_templates.append((path.name, tpl))
    if not rectangle_templates:
        print("All rectangle templates failed to load.")
        return

    for tpl, name in [(circle_tpl, "circle")] + [
            (t, n) for n, t in rectangle_templates]:
        th, tw = tpl.shape[:2]
        if tw > rw or th > rh:
            print(f"{name} template ({tw}x{th}) is larger than ROI "
                  f"({rw}x{rh}).")
            return

    stopper = _HoldToStop()

    print(f"Running detection loop. ROI={roi}. "
          f"{len(rectangle_templates)} rectangle template(s). "
          f"Stop with: Ctrl+C, hold '{STOP_HOLD_KEY}' for "
          f"{STOP_HOLD_SECONDS}s, or move mouse to a screen corner.")

    cycle = 0
    cycles_until_break = random.randint(BREAK_AFTER_MIN_CYCLES,
                                        BREAK_AFTER_MAX_CYCLES)

    try:
        while not stopper.check():
            shot = pyautogui.screenshot(region=roi)
            haystack = cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2GRAY)

            circle_matches = find_matches(haystack, circle_tpl, MATCH_THRESHOLD)
            picked_circle = (
                [random.choice(circle_matches)] if circle_matches else [])
            print(f"  circles:    {len(circle_matches)} match(es), "
                  f"clicking {len(picked_circle)}")
            _click_all(picked_circle, circle_tpl.shape, roi, stopper)
            if stopper.check():
                break

            for name, tpl in rectangle_templates:
                matches = find_matches(haystack, tpl, MATCH_THRESHOLD)
                print(f"  {name}: {len(matches)} match(es)")
                _click_all(matches, tpl.shape, roi, stopper)
                if stopper.check():
                    break
            if stopper.check():
                break

            cycle += 1
            if cycle >= cycles_until_break:
                pause = random.uniform(BREAK_MIN_SECONDS, BREAK_MAX_SECONDS)
                print(f"  -- break after {cycle} cycles, "
                      f"pausing {pause:.1f}s --")
                _sleep_with_check(pause, stopper)
                cycle = 0
                cycles_until_break = random.randint(BREAK_AFTER_MIN_CYCLES,
                                                    BREAK_AFTER_MAX_CYCLES)
            else:
                _sleep_with_check(random.uniform(MIN_DELAY, MAX_DELAY), stopper)
    except pyautogui.FailSafeException:
        print("Failsafe triggered. Exiting.")
        sys.exit(0)
    except KeyboardInterrupt:
        print("Stopped by user. Bye.")
```

- [ ] **Step 3: Run automated tests (regression check)**

Run: `python -m pytest tests/test_automouse.py -v`
Expected: 20 passed.

- [ ] **Step 4: Sanity-check that the module imports**

Run: `python -c "import automouse" && python -m py_compile automouse.py`
Expected: no output, exit 0.

- [ ] **Step 5: Commit**

```bash
git add automouse.py
git commit -m "feat: click loop iterates over all rectangle templates"
```

---

### Task 5: GUI — Listbox with Add/Delete replacing the single rectangle button

**Files:**
- Modify: `automouse.py` (rewrite the `App` class layout and add Listbox callbacks)

- [ ] **Step 1: Replace the App class**

Find the existing `class App:` block (its `__init__`, `on_set_roi`, `on_capture_circle`, `on_capture_rectangle`, `_capture_template_to`, `on_run`, and `refresh_status` methods) and replace the entire class with:

```python
class App:
    def __init__(self, root: tk.Tk) -> None:
        migrate_legacy_rectangle(TEMPLATES_DIR)

        self.root = root
        root.title("Automouse")
        root.geometry("320x420")

        tk.Button(root, text="Set ROI", width=32,
                  command=self.on_set_roi).pack(pady=4)
        tk.Button(root, text="Capture circle template", width=32,
                  command=self.on_capture_circle).pack(pady=4)

        tk.Label(root, text="Rectangle templates:",
                 anchor="w").pack(fill="x", padx=8, pady=(8, 0))

        list_frame = tk.Frame(root)
        list_frame.pack(fill="both", expand=False, padx=8)
        scrollbar = tk.Scrollbar(list_frame, orient="vertical")
        self.rect_list = tk.Listbox(list_frame, height=6,
                                    yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.rect_list.yview)
        scrollbar.pack(side="right", fill="y")
        self.rect_list.pack(side="left", fill="both", expand=True)

        btn_row = tk.Frame(root)
        btn_row.pack(pady=4)
        tk.Button(btn_row, text="+ Add rectangle",
                  command=self.on_add_rectangle).pack(side="left", padx=2)
        tk.Button(btn_row, text="Delete selected",
                  command=self.on_delete_rectangle).pack(side="left", padx=2)

        self.run_btn = tk.Button(root, text="Run", width=32,
                                 command=self.on_run)
        self.run_btn.pack(pady=8)

        self.status = tk.Label(root, text="", justify="left")
        self.status.pack(pady=4)

        self.refresh_status()

    def on_set_roi(self) -> None:
        result = _capture_rectangle(self.root)
        if result is None:
            return
        _, rect = result
        save_roi(CONFIG_PATH, rect)
        self.refresh_status()

    def on_capture_circle(self) -> None:
        result = _capture_rectangle(self.root)
        if result is None:
            return
        screenshot, (x, y, w, h) = result
        CIRCLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        screenshot.crop((x, y, x + w, y + h)).save(CIRCLE_PATH)
        self.refresh_status()

    def on_add_rectangle(self) -> None:
        result = _capture_rectangle(self.root)
        if result is None:
            return
        screenshot, (x, y, w, h) = result
        RECTANGLES_DIR.mkdir(parents=True, exist_ok=True)
        n = next_rectangle_number(RECTANGLES_DIR)
        path = RECTANGLES_DIR / f"{n:03d}.png"
        screenshot.crop((x, y, x + w, y + h)).save(path)
        self.refresh_status()

    def on_delete_rectangle(self) -> None:
        sel = self.rect_list.curselection()
        if not sel:
            return
        name = self.rect_list.get(sel[0])
        path = RECTANGLES_DIR / name
        if path.exists():
            path.unlink()
        self.refresh_status()

    def on_run(self) -> None:
        if not CONFIG_PATH.exists() or not CIRCLE_PATH.exists():
            messagebox.showerror("Automouse", "ROI or circle template missing.")
            return

        rect_paths = list_rectangle_templates(RECTANGLES_DIR)
        if not rect_paths:
            messagebox.showerror("Automouse",
                                 "Add at least one rectangle template.")
            return

        roi = load_roi(CONFIG_PATH)
        rx, ry, rw, rh = roi

        for path, name in [(CIRCLE_PATH, "circle")] + [
                (p, p.name) for p in rect_paths]:
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

    def refresh_status(self) -> None:
        rect_paths = list_rectangle_templates(RECTANGLES_DIR)

        self.rect_list.delete(0, tk.END)
        for p in rect_paths:
            self.rect_list.insert(tk.END, p.name)

        def mark(p: Path) -> str:
            return "OK" if p.exists() else "missing"
        self.status.config(text=(
            f"ROI:        {mark(CONFIG_PATH)}\n"
            f"Circle:     {mark(CIRCLE_PATH)}\n"
            f"Rectangles: {len(rect_paths)}"
        ))
        all_ready = (CONFIG_PATH.exists()
                     and CIRCLE_PATH.exists()
                     and len(rect_paths) >= 1)
        self.run_btn.config(state=tk.NORMAL if all_ready else tk.DISABLED)
```

- [ ] **Step 2: Run automated tests (regression check)**

Run: `python -m pytest tests/test_automouse.py -v`
Expected: 20 passed.

- [ ] **Step 3: Sanity-check imports / compile**

Run: `python -c "import automouse" && python -m py_compile automouse.py`
Expected: no output, exit 0.

- [ ] **Step 4: Commit**

```bash
git add automouse.py
git commit -m "feat: GUI list of rectangle templates with Add/Delete"
```

---

### Task 6: End-to-end smoke test (manual)

**Files:** none modified.

This is verified by the user, since the GUI needs a display and the click loop drives the cursor. Document the steps so the operator knows what to check.

- [ ] **Step 1: Migration of legacy rectangle.png**

Set up: ensure `templates/rectangle.png` exists from older runs and `templates/rectangles/` does not.

Run: `python automouse.py`

Expected:
- Window opens, status shows `Rectangles: 1`.
- The Listbox shows `001.png`.
- `templates/rectangle.png` no longer exists.
- `templates/rectangles/001.png` exists with the original bytes.

- [ ] **Step 2: Add and delete templates**

In the running GUI:

1. Click **+ Add rectangle**, drag a rectangle on screen, release.
2. Status updates to `Rectangles: 2`. Listbox now shows `001.png`, `002.png`.
3. Add another → `Rectangles: 3`, Listbox shows `001.png`, `002.png`, `003.png`.
4. Select `002.png` and click **Delete selected**. Status shows `Rectangles: 2`. Listbox shows `001.png`, `003.png`. Disk reflects the same.
5. Add a new template → it should be `004.png` (max + 1), not refilling the gap.

- [ ] **Step 3: Run the loop with multiple templates**

Click **Run**.

Expected:
- Terminal prints
  `Running detection loop. ROI=... 3 rectangle template(s). ...`.
- Each cycle prints `circles: ... clicking ...` followed by one line per
  rectangle template (`001.png: N match(es)`, `003.png: N match(es)`,
  `004.png: N match(es)`).
- Hold `s` for 2s — loop stops with the existing message.

- [ ] **Step 4: Run via CLI without re-capturing**

Run: `python automouse.py run`
Expected: same loop starts immediately, same per-template logging.

- [ ] **Step 5: Validation paths**

1. Capture a tiny ROI (e.g. 30×30) then attempt to add a rectangle template
   larger than that ROI. When you click **Run**, the messagebox should say
   "<filename> template (WxH) is larger than ROI (30x30)..." — naming the
   offender. Re-capture or enlarge the ROI to recover.
2. Delete every rectangle template; the **Run** button should disable.

If all five steps pass, the feature is verified.

---

## Self-Review Notes

- **Spec coverage:**
  - Storage layout (`templates/rectangles/NNN.png`) → Task 4 (constant) + Task 5 (writes via Add).
  - Numeric allocation (`max + 1`, gaps left in place) → Task 2 + Task 5 `on_add_rectangle`.
  - Migration of legacy `rectangle.png` → Task 3 + wired in Task 4 (loop) and Task 5 (`App.__init__`).
  - GUI Listbox + Add/Delete + status row + Run-disabled-when-empty → Task 5.
  - Click loop iterating all templates per cycle → Task 4.
  - Per-template ROI-size validation → Task 4 (loop) + Task 5 (Run button).
  - Unit tests for the three pure helpers → Tasks 1–3.
  - Smoke test → Task 6.

- **Placeholders:** none. Every code block is complete.

- **Type/name consistency:** `RECTANGLES_DIR`, `list_rectangle_templates`, `next_rectangle_number`, `migrate_legacy_rectangle` are used identically in every reference. The click-loop loops over `(name, tpl)` pairs in both Task 4 (validation) and the cycle body — consistent shape `Tuple[str, np.ndarray]`.

- **One follow-up:** the `pre-flight` validation in Task 4 reuses the
  same logic as the GUI's `on_run` in Task 5. That duplication is
  intentional — the CLI `run` mode skips the GUI entirely, so it needs
  its own pre-flight. Refactoring to share is YAGNI for this scope.
