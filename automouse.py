"""Template-matching auto-clicker — see docs/superpowers/specs/2026-05-01-template-matching-clicker-design.md"""

import cv2
import json
import math
import numpy as np
import pyautogui
import random
import re
import sys
import time
import tkinter as tk
from difflib import SequenceMatcher
from pathlib import Path
from PIL import Image
from Quartz import (
    CGEventSourceKeyState,
    CGWarpMouseCursorPosition,
    kCGEventSourceStateHIDSystemState,
)
from tkinter import messagebox
from typing import Dict, List, Optional, Tuple

TEMPLATES_DIR = Path("templates")
CONFIG_PATH = TEMPLATES_DIR / "config.json"
CIRCLE_PATH = TEMPLATES_DIR / "circle.png"
RECTANGLES_DIR = TEMPLATES_DIR / "rectangles"
ANSWERS_PATH = TEMPLATES_DIR / "answers.json"
BLUE_SAMPLE_PATH = TEMPLATES_DIR / "blue_sample.png"

MATCH_THRESHOLD = 0.8
MIN_DELAY = 0.17
MAX_DELAY = 0.33
STOP_HOLD_KEY = "s"
STOP_HOLD_KEYCODE = 1  # macOS virtual keycode for 's' (kVK_ANSI_S)
STOP_HOLD_SECONDS = 2.0
STOP_POLL_INTERVAL = 0.05

# Human-like cursor movement: cubic Bezier path from current pos to target,
# with random perpendicular curvature, per-step jitter, and small random pauses.
MOVE_PIXELS_PER_STEP = 36       # path resolution
MOVE_STEP_MIN_DELAY = 0.002     # seconds between intermediate moves
MOVE_STEP_MAX_DELAY = 0.005
MOVE_CURVE_STRENGTH = 0.18      # max perpendicular offset as fraction of distance
MOVE_JITTER_PIXELS = 0.7        # stddev of per-step Gaussian jitter

# Long breaks between bursts of cycles, to look more human.
BREAK_AFTER_MIN_CYCLES = 10
BREAK_AFTER_MAX_CYCLES = 30
BREAK_MIN_SECONDS = 5.0
BREAK_MAX_SECONDS = 10.0

# Question/answer memory feature.
BLUE_COLOR_TOLERANCE = 25
QUESTION_MATCH_THRESHOLD = 0.9
ANSWER_MATCH_THRESHOLD = 0.9

ROI = Tuple[int, int, int, int]  # (x, y, width, height) in screen pixels
BBox = Tuple[int, int, int, int]  # (x, y, w, h) in pixels, top-left origin


def save_roi(config_path: Path, roi: ROI) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps({"roi": list(roi)}))


def load_roi(config_path: Path) -> Optional[ROI]:
    if not config_path.exists():
        return None
    data = json.loads(config_path.read_text())
    x, y, w, h = data["roi"]
    return (x, y, w, h)


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


_RECT_NAME_RE = re.compile(r"^(\d{3})\.png$")


def list_rectangle_templates(directory: Path) -> List[Path]:
    """Return zero-padded NNN.png files under directory, sorted by name."""
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and _RECT_NAME_RE.match(p.name)
    )


def next_rectangle_number(directory: Path) -> int:
    """Return max existing NNN integer + 1, or 1 if none."""
    paths = list_rectangle_templates(directory)
    if not paths:
        return 1
    nums = [int(_RECT_NAME_RE.match(p.name).group(1)) for p in paths]
    return max(nums) + 1


def migrate_legacy_rectangle(templates_dir: Path) -> None:
    """If templates_dir/rectangle.png exists and rectangles/ does not,
    move the legacy file to rectangles/001.png. Idempotent."""
    legacy = templates_dir / "rectangle.png"
    new_dir = templates_dir / "rectangles"
    if not legacy.exists() or new_dir.exists():
        return
    new_dir.mkdir(parents=True)
    legacy.rename(new_dir / "001.png")


def normalize_ocr_text(observations: List[Tuple[str, BBox]]) -> str:
    """Lowercase, strip, drop blanks, dedupe, sort, join with newlines.
    Order-independent so shuffled answers don't change the signature."""
    lines = sorted({t.strip().lower()
                    for t, _ in observations
                    if t.strip()})
    return "\n".join(lines)


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


def color_mask(image: Image.Image, target_rgb: Tuple[int, int, int],
               tolerance: int = BLUE_COLOR_TOLERANCE) -> np.ndarray:
    """Return an HxW bool mask of pixels within `tolerance` per channel
    of target_rgb."""
    arr = np.asarray(image.convert("RGB"))
    diff = np.abs(arr.astype(int) - np.asarray(target_rgb, dtype=int))
    return np.all(diff <= tolerance, axis=-1)


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


def dominant_color(image: Image.Image) -> Tuple[int, int, int]:
    """Return the median (R, G, B) of all pixels."""
    arr = np.asarray(image.convert("RGB"))
    flat = arr.reshape(-1, 3)
    r, g, b = np.median(flat, axis=0).astype(int).tolist()
    return (int(r), int(g), int(b))


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


class _HoldToStop:
    """Polls macOS HID; sets `stop` to True when STOP_HOLD_KEY is held for
    STOP_HOLD_SECONDS. Must be polled (.check()) from the main thread —
    pynput-style listener threads crash on recent macOS because TSM APIs
    require the main thread."""

    def __init__(self) -> None:
        self.stop = False
        self._press_started: Optional[float] = None

    def check(self) -> bool:
        if CGEventSourceKeyState(kCGEventSourceStateHIDSystemState,
                                 STOP_HOLD_KEYCODE):
            now = time.monotonic()
            if self._press_started is None:
                self._press_started = now
            elif now - self._press_started >= STOP_HOLD_SECONDS:
                self.stop = True
        else:
            self._press_started = None
        return self.stop


def _sleep_with_check(seconds: float, stopper: _HoldToStop) -> None:
    """Like time.sleep, but polls the stopper every STOP_POLL_INTERVAL."""
    deadline = time.monotonic() + seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or stopper.check():
            return
        time.sleep(min(STOP_POLL_INTERVAL, remaining))


def _move_cursor(x: float, y: float) -> None:
    """Visibly move the cursor on macOS. pyautogui.moveTo only posts a
    kCGEventMouseMoved event — that updates the system's logical mouse
    position (so clicks land correctly) but doesn't move the visible
    cursor on modern macOS. CGWarpMouseCursorPosition does. Call both so
    the cursor is visible AND apps receive a moved event."""
    CGWarpMouseCursorPosition((x, y))
    pyautogui.moveTo(x, y, _pause=False)


def _human_move_to(x: int, y: int) -> None:
    """Move the cursor to (x, y) along a cubic-Bezier curve with jitter,
    instead of jumping straight to the target. Helps mask automation."""
    sx, sy = pyautogui.position()
    dx, dy = x - sx, y - sy
    distance = math.hypot(dx, dy)
    if distance < 3:
        _move_cursor(x, y)
        return

    perp_x, perp_y = -dy / distance, dx / distance
    o1 = random.uniform(-MOVE_CURVE_STRENGTH, MOVE_CURVE_STRENGTH) * distance
    o2 = random.uniform(-MOVE_CURVE_STRENGTH, MOVE_CURVE_STRENGTH) * distance
    cp1x = sx + dx * 0.3 + perp_x * o1
    cp1y = sy + dy * 0.3 + perp_y * o1
    cp2x = sx + dx * 0.7 + perp_x * o2
    cp2y = sy + dy * 0.7 + perp_y * o2

    steps = max(10, int(distance / MOVE_PIXELS_PER_STEP))
    for i in range(1, steps):
        # Ease-in-out so velocity ramps up and down, not constant.
        t = (1 - math.cos(math.pi * i / steps)) / 2
        u = 1 - t
        bx = u**3 * sx + 3*u**2*t * cp1x + 3*u*t**2 * cp2x + t**3 * x
        by = u**3 * sy + 3*u**2*t * cp1y + 3*u*t**2 * cp2y + t**3 * y
        bx += random.gauss(0, MOVE_JITTER_PIXELS)
        by += random.gauss(0, MOVE_JITTER_PIXELS)
        _move_cursor(bx, by)
        time.sleep(random.uniform(MOVE_STEP_MIN_DELAY, MOVE_STEP_MAX_DELAY))
    _move_cursor(x, y)


def _click_all(matches: List[Tuple[int, int]],
               template_shape: Tuple[int, int],
               roi: ROI,
               stopper: Optional[_HoldToStop] = None) -> None:
    """Click each match center with a random delay between clicks."""
    th, tw = template_shape[:2]
    rx, ry, _, _ = roi
    for mx, my in matches:
        if stopper is not None and stopper.check():
            return
        cx = rx + mx + tw // 2
        cy = ry + my + th // 2
        _human_move_to(cx, cy)
        pyautogui.click()
        delay = random.uniform(MIN_DELAY, MAX_DELAY)
        if stopper is not None:
            _sleep_with_check(delay, stopper)
        else:
            time.sleep(delay)


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


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        run_detection_loop()
        return
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
