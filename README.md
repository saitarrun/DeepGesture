# Deep Gesture

Turn a standard webcam into a pointing device. Live video is run through
[MediaPipe](https://developers.google.com/mediapipe) to track 21 hand
landmarks, gesture logic interprets fingertip positions, and
[PyAutoGUI](https://pyautogui.readthedocs.io/) injects the resulting cursor
moves, clicks, and scrolls into the OS. No special hardware required.

This is **Phase 4 — accuracy-hardened**: a full gesture set (move, click, drag,
right click, scroll), speed-adaptive cursor smoothing, system-control gestures
(volume, screenshot, Mission Control, Spaces), custom ML gesture training, a
threaded capture pipeline, and an industry-grade accuracy layer (see
[Accuracy engineering](#accuracy-engineering)). The tracker uses MediaPipe's
modern **Tasks API** (`hand_landmarker.task` model, included). The on-screen
keyboard from the original blueprint is still deferred.

## Setup

```bash
cd "Hand Gesture"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### macOS permissions (required)

PyAutoGUI cannot move the cursor and OpenCV cannot open the camera until the
app running Python is granted permission. In **System Settings → Privacy &
Security**, add your terminal (e.g. Terminal.app / iTerm) to **both**:

- **Camera**
- **Accessibility**

Quit and relaunch the terminal after granting, or the change won't take effect.

## Run

```bash
python virtual_mouse.py
# options:
python virtual_mouse.py --camera 1 --width 1920 --height 1080 --fps 30 -v
```

| Flag | Meaning |
| --- | --- |
| `--camera N` | camera index (default 0) |
| `--width / --height / --fps` | capture format (default 1280×720@60) |
| `--model PATH` | alternate `hand_landmarker.task` |
| `--hud neon\|classic` | overlay style (default: neon) |
| `-v / --verbose` | debug logging |

A mirrored webcam window opens with a HUD.

## The neon HUD

The default overlay (`neon_hud.py`) is a cyberpunk, **gesture-reactive** UI —
every element is driven by the same state the gesture engine uses, so the
interface *is* the feedback loop. The whole UI re-tints to the active pose's
color (cyan = cursor, yellow = scroll, orange = swipe, magenta = volume,
green = ML, …) — and the tint **crossfades** between poses instead of
snapping.

**Cyberpunk layer & animation engine:**

- **CRT ambience** — the camera image is graded through a duotone color LUT
  (lifted blues, crushed greens), with scanlines, a corner vignette, and a
  scan sweep slowly roaming down the frame. Cached overlays: ~0.3 ms total.
- **Boot sequence** — on startup a reveal line sweeps across the screen while
  `VMOUSE://CYBERDECK v6.0 -- SYSTEM ONLINE` types itself out.
- **Glitch bursts** — every action that fires on the bus (click, swipe, menu
  pick…) triggers a ~120 ms datamosh: shifted row bands + RGB channel split.
  Pose flips get a shorter burst.
- **Chromatic aberration** on the pose title (blue/red ghost passes).
- **Everything eases** — a frame-rate-independent exponential tween drives the
  accent crossfade, pinch-gauge and swipe-charge fills, pie-menu hover heat
  (slices warm up and cool down), the legend highlight bar (slides between
  rows), and cubic ease-out toast slide-ins. No element ever snaps.

**Gesture-reactive elements:**

- **Glowing hand skeleton** — neon strokes rendered to a separate layer,
  blurred at quarter resolution and added back (a real glow, ~2 ms/frame).
- **Pinch-charge ring** on the index fingertip — fills as thumb and index
  close; a full ring is exactly the click threshold, so you *see* the click
  coming. Pulses green while pinched, turns amber with a `DRAG` tag while
  dragging. A rotating two-arc reticle spins around the fingertip.
- **Motion trail** — fading streak behind the cursor finger.
- **Swipe energy bar** — under an open palm, a bar charges toward the trigger
  distance in the direction you sweep, with marching chevrons; it flashes
  green at the commit point.
- **Animated active region** — corner brackets plus marching dashes instead of
  a plain rectangle.
- **Status bar with pose notch** — the current pose in big type front and
  center, live FPS sparkline on the left, inference/loop/jitter and tuning
  values on the right.
- **Gesture map** — the legend highlights the row of whatever pose you are
  holding, in that pose's color.
- **ML readout** — label + confidence bar when a custom gesture is active.
- **Training mode** — pulsing red REC border with per-class sample chips;
  **calibration** gets an animated crosshair.

`--hud classic` brings back the plain overlay. The HUD is covered by headless
render tests (every pose + overlays) in the test suite. The orange rectangle is the
**active region** — your index fingertip inside it maps to the full screen.

## Gestures

| Hand pose | Action |
| --- | --- |
| Index finger up (only) | Move cursor (1€ adaptive smoothing) |
| ↳ thumb + index pinch (quick tap) | Left click — two quick taps = double-click |
| ↳ thumb + index pinch (hold + move) | Drag & drop |
| Index + middle, tips **apart** | Scroll — vertical **and** horizontal |
| Index + middle, tips **touching** | Right click |
| Three fingers (index+middle+ring) | Volume — move hand up / down |
| Index + pinky (🤘 rock) | Mission Control |
| Pinky only | Screenshot (⌘⇧3) |
| **Two hands**, both pinching, move apart / together | **Zoom in / out** (⌘+ / ⌘−) |
| **Open palm, swipe sideways** | **Switch desktop / Space** (Ctrl ← / Ctrl →) |
| Rock 🤘 / pinky | **Macros** (configurable — see below) |
| Fist | Neutral (pause; reposition freely) |
| **Fist, held ~0.8 s** | **Radial pie menu** (point + pinch to fire) |
| Cursor into any screen corner | FAILSAFE abort (PyAutoGUI) |
| `q` in the window | Quit |

## Calibration & profile

Press **`c`** to start calibration, move your index finger to the comfortable
extremes of your reach, then press **`c`** again. The active region (the box)
snaps to your hand's range so the whole screen is reachable without stretching.

Everything persists to **`config.json`** (auto-created on first run, saved on
quit): your smoothing values (`min_cutoff`, `beta`), the calibration box, and
the macro table.

## Macros

`config.json` → `"macros"` maps a hand pose to a hotkey passed to
`pyautogui.hotkey()`. Defaults:

```json
"macros": {
  "rock":  ["ctrl", "up"],            // Mission Control
  "pinky": ["command", "shift", "3"]  // Screenshot
}
```

Bindable pose names: `rock`, `pinky`, `fist`. (`palm` is reserved for the swipe
gesture.) Edit the keys to anything `pyautogui` understands, e.g.
`["command", "c"]` for copy. Macros fire once per gesture (debounced).

## Switching desktops (Spaces) by swipe

Hold an **open palm** and sweep it left or right — the app fires **Ctrl + ←/→**
to move between macOS Spaces. Two requirements on the OS side:

1. **More than one desktop** must exist (add Spaces in Mission Control).
2. **System Settings → Keyboard → Keyboard Shortcuts → Mission Control →**
   "Move left/right a space" must be **enabled** (they are by default, but some
   setups disable them).

Tune feel via `SWIPE_DIST_FRAC` (travel needed to trigger, as a fraction of
frame width) and `SWIPE_WINDOW` (how fast the swipe must be) near the top of
`virtual_mouse.py`.

## Radial pie menu

Hold a **fist** for ~0.8 s and a neon donut menu blooms around your hand.
**Point** your index finger at a slice (it highlights), **pinch** to fire it,
make a **fist again** (or wait 5 s) to close. All other gestures are suspended
while the menu is open, so it can never mis-click.

Slices live in `config.json → "menu"` — any label + hotkey:

```json
"menu": [
  {"label": "Copy", "keys": ["command", "c"]},
  {"label": "Spotlight", "keys": ["command", "space"]},
  ...
]
```

Defaults: Copy, Paste, Screenshot, Mission Control, Spotlight, Undo.

## Action toasts (the consistency layer)

Every OS side effect — click, drag, scroll, swipe, zoom, volume, macro, ML
gesture, menu pick, profile switch — flows through a single **ActionBus**
(`actions.py`). The bus owns one cooldown table (no scattered debounce timers)
and records every fire as an event; the HUD renders those events as **toasts**
in the top-right (slide in, fade out). Continuous actions aggregate into one
rolling toast (`SCROLL +14`) instead of spamming. What you see fired is, by
construction, exactly what fired.

## Per-app profiles

A background watcher (`app_profiles.py`) polls the frontmost macOS app once a
second and hot-swaps gesture bindings. Overridable per app: the palm-swipe
hotkeys, the pose macros, and the ML `gesture_actions`. Example (shipped by
default — in Chrome, palm-swipe switches **tabs** instead of Spaces):

```json
"app_profiles": {
  "Google Chrome": {
    "swipe": {"next": ["command", "shift", "]"],
              "prev": ["command", "shift", "["]}
  }
}
```

The active profile shows as a chip in the top bar, and switching apps pops a
`PROFILE` toast. Note: frontmost-app detection uses System Events — macOS may
show a one-time **Automation** permission prompt; if denied, the app quietly
stays on the default profile.

## Custom gesture training (ML)

The app ships a tiny KNN classifier (`gesture_ml.py`) that learns gestures you
record yourself. Features are translation-, scale-, and rotation-invariant, so
a trained gesture is recognized anywhere in frame, at any hand size or angle.

**Train:**
1. Press **`t`** to enter training mode.
2. Hold a pose and press a **number `1`–`9`** to add a sample for class `gN`.
   Move your hand slightly and press again — ~10–15 samples per class is plenty.
3. Press **`t`** again to finish. Samples persist in `gestures.json`.

**Recognize / bind:** a recognized class fires the hotkey bound to it in
`config.json` → `"gesture_actions"` (defaults: `g1` = copy, `g2` = paste). To
avoid fighting the built-in controls, ML gestures only act while your hand is in
the **"other" pose** (anything that isn't index/two/three/palm/rock/pinky/fist).
The live prediction and distance show on the HUD; press **`[`** / **`]`** to
lower / raise the match threshold (lower = stricter).

To reset training, delete `gestures.json`.

## Performance (threaded pipeline)

`threaded_capture.py` runs the camera on its own thread holding only the latest
frame, so capture I/O overlaps with MediaPipe inference and stale frames are
dropped instead of queued — higher effective FPS, lower latency. The HUD's
second line shows live **FPS**, **inference ms**, **loop ms**, and frame-time
**jitter**.

## Tuning

All constants live at the top of `virtual_mouse.py`:

| Constant | Meaning |
| --- | --- |
| `EURO_MIN_CUTOFF` | Lower = smoother when the hand is still |
| `EURO_BETA` | Higher = snappier (less lag) when the hand moves fast |
| `PINCH_ON` / `PINCH_OFF` | Pinch engage / release ratios (hysteresis dead band) |
| `RCLICK_TOUCH_RATIO` | Index–middle gap (× hand scale) that triggers right click |
| `FINGER_EXT_RATIO` | How much farther than its PIP a tip must be to count "up" |
| `MIN_HAND_SCORE` | Confidence below which a detected hand is ignored |
| `DRAG_HOLD_SEC` | Pinch held longer than this starts a drag, not a click |
| `FRAME_MARGIN` | Size of the inactive border; smaller = more reach needed |
| `SCROLL_SENSITIVITY` | Multiplier on hand motion → scroll amount |
| `VOL_STEP_FRAC` / `VOLUME_DELTA` | Hand travel per volume step (fraction of hand scale, accumulates) / percent per step |

### Live tuning (while the app runs)

| Key | Effect |
| --- | --- |
| `a` / `z` | Lower / raise `min_cutoff` — steadiness when the hand is still |
| `s` / `x` | Lower / raise `beta` — responsiveness when the hand moves fast |

The current values show in the top-right of the HUD. Smoother-but-laggier:
lower both. Snappier: raise `beta`. Find your feel, then copy the values into
the constants at the top of `virtual_mouse.py`.

## Accuracy engineering

What makes the recognition robust rather than demo-grade:

- **Scale-invariant thresholds** — every geometric threshold (pinch, right-click
  touch) is a *ratio of the hand's own scale* (wrist → middle-MCP distance),
  not a pixel count. A pinch behaves identically at 40 cm and at 1.5 m from the
  camera. Swipe thresholds are fractions of the frame width, so they survive
  resolution changes.
- **Pinch hysteresis (Schmitt trigger)** — the pinch must close below
  `PINCH_ON` (0.40) to engage and re-open above `PINCH_OFF` (0.55) to release.
  The dead band between them means noise at the boundary can never flicker the
  state — no phantom double clicks. The ratio itself is EMA-smoothed first.
- **Rotation-invariant finger detection** — a finger counts as extended when
  its tip is farther from the wrist than its PIP joint (distance-based), so
  poses classify correctly with the hand tilted 40°+, where the naive
  "tip above PIP" y-test breaks.
- **Sub-pixel landmarks** — landmark coordinates stay floating-point end to
  end; nothing is quantized to integer pixels before the smoothing filter, so
  the cursor resolves finer than one camera pixel.
- **Real timestamps into MediaPipe** — `detect_for_video()` receives true
  monotonic milliseconds (strictly increasing), so MediaPipe's internal
  temporal tracking sees actual frame spacing instead of a fake fixed-step
  counter.
- **Per-hand confidence gating** — hands below `MIN_HAND_SCORE` confidence are
  discarded before any gesture logic runs.
- **Fractional scroll accumulation** — slow, precise scroll motion accumulates
  fractional ticks instead of being truncated to zero each frame; the deadzone
  still rejects jitter.
- **Cursor freeze on pinch** — the cursor locks the instant you pinch, so a
  click never drifts as your finger closes. It only unlocks for a deliberate
  hold-and-drag.
- **Pose debouncing** — a pose must hold for a few frames before it commits,
  killing mode flicker (no accidental scrolls/clicks from a twitching finger).
- **Outlier clamp** — single-frame landmark "teleports" are rejected.
- **1€ adaptive cursor filter** — see below.
- **720p @ 60 fps capture** — finer landmarks and more samples for the filter.

## Reliability & tests

- Structured logging (`-v` for debug), CLI flags, and graceful handling of the
  PyAutoGUI FAILSAFE (corner abort exits cleanly, releasing any held drag).
- **`test_virtual_mouse.py`** — a 19-test headless regression suite (PyAutoGUI
  is stubbed; synthetic 21-landmark hands drive the real gesture code). It
  covers pose classification under rotation, pinch scale-invariance and
  hysteresis, click/drag state machine, scroll accumulation + deadzone, swipe
  accept/reject, 1€ filter behaviour, ML feature invariance and KNN
  separation, and config round-trips:

```bash
python test_virtual_mouse.py
```

### How the smoothing works (1€ filter)

Instead of a fixed smoothing factor, the cursor uses a **One-Euro filter** — an
adaptive low-pass filter. It estimates how fast the fingertip is moving and
raises the cutoff frequency with speed:

```
cutoff = min_cutoff + beta * |velocity|
```

When your hand is nearly still, the cutoff is low so jitter is heavily damped
(rock-steady cursor). When you move quickly, the cutoff rises so the filter
follows almost instantly (no lag). This is why it feels both stable *and*
responsive — a fixed exponential-moving-average can only be one or the other.
Tune `EURO_MIN_CUTOFF` (steadiness when still) and `EURO_BETA` (responsiveness
when moving).

## Roadmap

- On-screen virtual keyboard / typing
- Two-handed gestures (pinch-zoom, rotate)
- Per-user calibration UI and config file
```
 and config file
```
