"""Template-matching auto-clicker — see docs/superpowers/specs/2026-05-01-template-matching-clicker-design.md"""

import cv2
import json
import math
import numpy as np
import pyautogui
import random
import sys
import time
import tkinter as tk
from pathlib import Path
from PIL import Image
from Quartz import (
    CGEventSourceKeyState,
    CGWarpMouseCursorPosition,
    kCGEventSourceStateHIDSystemState,
)
from tkinter import messagebox
from typing import List, Optional, Tuple

TEMPLATES_DIR = Path("templates")
CONFIG_PATH = TEMPLATES_DIR / "config.json"
CIRCLE_PATH = TEMPLATES_DIR / "circle.png"
RECTANGLE_PATH = TEMPLATES_DIR / "rectangle.png"

MATCH_THRESHOLD = 0.8
MIN_DELAY = 0.5
MAX_DELAY = 1.0
STOP_HOLD_KEY = "s"
STOP_HOLD_KEYCODE = 1  # macOS virtual keycode for 's' (kVK_ANSI_S)
STOP_HOLD_SECONDS = 2.0
STOP_POLL_INTERVAL = 0.05

# Human-like cursor movement: cubic Bezier path from current pos to target,
# with random perpendicular curvature, per-step jitter, and small random pauses.
MOVE_PIXELS_PER_STEP = 12       # path resolution
MOVE_STEP_MIN_DELAY = 0.005     # seconds between intermediate moves
MOVE_STEP_MAX_DELAY = 0.015
MOVE_CURVE_STRENGTH = 0.18      # max perpendicular offset as fraction of distance
MOVE_JITTER_PIXELS = 0.7        # stddev of per-step Gaussian jitter

# Long breaks between bursts of cycles, to look more human.
BREAK_AFTER_MIN_CYCLES = 10
BREAK_AFTER_MAX_CYCLES = 30
BREAK_MIN_SECONDS = 5.0
BREAK_MAX_SECONDS = 10.0

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

    stopper = _HoldToStop()

    print(f"Running detection loop. ROI={roi}. Stop with: Ctrl+C, "
          f"hold '{STOP_HOLD_KEY}' for {STOP_HOLD_SECONDS}s, or move "
          f"mouse to a screen corner.")

    cycle = 0
    cycles_until_break = random.randint(BREAK_AFTER_MIN_CYCLES,
                                        BREAK_AFTER_MAX_CYCLES)

    try:
        while not stopper.check():
            shot = pyautogui.screenshot(region=roi)
            haystack = cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2GRAY)

            circle_matches = find_matches(haystack, circle_tpl, MATCH_THRESHOLD)
            picked_circle = [random.choice(circle_matches)] if circle_matches else []
            print(f"  circles:    {len(circle_matches)} match(es), "
                  f"clicking {len(picked_circle)}")
            _click_all(picked_circle, circle_tpl.shape, roi, stopper)
            if stopper.check():
                break

            rect_matches = find_matches(haystack, rect_tpl, MATCH_THRESHOLD)
            print(f"  rectangles: {len(rect_matches)} match(es)")
            _click_all(rect_matches, rect_tpl.shape, roi, stopper)
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
        print(f"'{STOP_HOLD_KEY}' held for "
              f"{STOP_HOLD_SECONDS}s — stopping.")
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
