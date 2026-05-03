# automouse

Template-matching auto-clicker for macOS. Originally a random mouse mover to
keep the Mac awake; now also a quiz-bot that recognizes questions, learns
correct answers via OCR, and clicks them on subsequent encounters.

## What's in this repo

| File | Purpose |
| --- | --- |
| `automouse.py` | Main app: Tkinter GUI for setup, click loop with OCR memory and end-of-batch flow. |
| `random_mouse_mover.py` | Standalone idle-keeper. Independent of `automouse.py`. |
| `tests/test_automouse.py` | 63 unit tests covering pure helpers. |
| `templates/` | Runtime data (auto-created): ROI, captured templates, learned answers. **gitignored**. |
| `docs/superpowers/` | Specs and implementation plans for each feature. |

## Setup

```bash
pip install -r requirements.txt
```

macOS permissions (System Settings → Privacy & Security):

- **Screen Recording** — for `pyautogui.screenshot` and Apple Vision OCR.
- **Accessibility** — for `pyautogui.click`/`moveTo` and the global hold-`s` stop.

Both must be granted to the terminal/launcher running Python.

## Quick start

```bash
python automouse.py
```

In the GUI:

1. **Set ROI** — drag a rectangle around the screen area to monitor.
2. **Capture circle template** — drag a tight box around an answer-option
   radio button (a "circle").
3. **+ Add rectangle** — drag a tight box around the Next/Tiếp button.
4. **Run** — closes the main window, opens a small "running" window with a
   Stop button and a live log.

The bot now loops:

- Screenshot the ROI → OCR for the question.
- If the question signature matches one in `answers.json` (≥ 0.9 fuzzy),
  click the stored answer text directly.
- Otherwise click a random circle, wait 0.6 s for the app to reveal the
  correct answer, and learn the `(question, answer)` pair from the line
  below the anchor phrase (default: `"Câu trả lời chính xác là"`).
- Click the Next/Tiếp button to advance.

## CLI run mode

`python automouse.py run` skips the GUI and starts the loop directly using
the saved templates and ROI.

## Stopping the loop

- **Stop button** in the running window.
- **Hold `s` for 2 s** anywhere on screen.
- **Move the mouse hard to a screen corner** (pyautogui failsafe).
- `Ctrl+C` in the terminal.

When stopped, the running window closes and the main GUI returns with
updated counts so you can adjust templates and re-run.

## Question / answer memory

The bot stores `(question, answer)` pairs in `templates/answers.json`.

- **Question signature** = the longest OCR observation containing `?` on
  the current screen, lowercased. Stable across attempts because it ignores
  navigation chrome and timestamps.
- **Answer extraction** = OCR the screen, find the line matching the
  *answer anchor* phrase (configurable via "Set answer anchor phrase";
  defaults to `Câu trả lời chính xác là`), then take the closest text
  observation directly below it. Skips navigation words like
  `Trước`/`Tiếp`/`OK`/`Hủy` to avoid poisoning the memory.

The live log on the Stop window shows each event:

```
[memory] LEARNED: '2-Biển 2.'  ◀ Q: Biển nào báo hiệu "Giao nhau với đường ưu tiên"?
[memory] recognized → click '2-Biển 2.'  ◀ Q: ...
[memory] anchor not visible; skipping store  ◀ Q: ...
```

## End-of-batch flow

When the primary rectangle (e.g. Tiếp) is no longer visible — typically at
the end of a quiz batch — the bot can run a configured *recovery sequence*
instead of looping fruitlessly. In the GUI under **End-of-batch flow**:

1. **+ Add end-flow step** for each step in order. For the Vietnamese
   driving-test app: `001` = Kết thúc luyện thi, `002` = OK, `003` = Luyện
   tất cả.

When Tiếp returns 0 matches in a cycle, the bot screenshots, finds the
first end-flow step, clicks it, waits 1.2 s, screenshots again, finds the
next step, clicks, repeats. Aborts cleanly if any step's template is not
visible.

End-flow is optional; leaving the list empty disables it.

## Anti-detection touches

- Each click is preceded by a randomized cubic-Bézier mouse path with
  per-waypoint Gaussian jitter and ease-in-out timing.
- The cursor is warped via `Quartz.CGWarpMouseCursorPosition` so the
  movement is visible (pyautogui alone only updates the system position).
- Inter-click delays are random in [0.17, 0.33] s.
- Every 10–30 cycles, the bot pauses 5–10 s.
- Click target is randomized within each cycle (e.g. one random circle).

Tweak the `MIN_DELAY`, `MAX_DELAY`, `BREAK_*`, and `MOVE_*` constants near
the top of `automouse.py` to adjust pacing.

## On-disk layout

```
templates/
├── config.json              # ROI, optional blue_rgb (legacy), optional answer_anchor
├── circle.png               # circle (radio button) template
├── rectangles/              # primary rectangles (Tiếp / Next)
│   ├── 001.png
│   └── ...
├── end_flow/                # end-of-batch sequence (in click order)
│   ├── 001.png
│   ├── 002.png
│   └── 003.png
├── answers.json             # learned [{"question": ..., "answer": ...}]
└── blue_sample.png          # legacy blue-line sample (unused)
```

The `templates/` directory is gitignored — every install builds its own.

## Running the tests

```bash
python -m pytest tests/test_automouse.py -v
```

Apple Vision OCR is integration-tested manually (see Task 10 in
`docs/superpowers/plans/2026-05-01-question-answer-memory.md`).

## Project history

This repo was built incrementally through brainstormed specs and TDD'd
plans. See `docs/superpowers/specs/` and `docs/superpowers/plans/` for the
design trail of each feature.
