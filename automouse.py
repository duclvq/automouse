"""Template-matching auto-clicker — see docs/superpowers/specs/2026-05-01-template-matching-clicker-design.md"""

import cv2
import json
import math
import numpy as np
import pyautogui
import time
import tkinter as tk
from pathlib import Path
from PIL import Image
from tkinter import messagebox
from typing import List, Optional, Tuple

TEMPLATES_DIR = Path("templates")
CONFIG_PATH = TEMPLATES_DIR / "config.json"
CIRCLE_PATH = TEMPLATES_DIR / "circle.png"
RECTANGLE_PATH = TEMPLATES_DIR / "rectangle.png"

MATCH_THRESHOLD = 0.8
MIN_DELAY = 0.5
MAX_DELAY = 1.0

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


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
