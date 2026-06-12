"""
Hand-Gesture Virtual Mouse (Phase 4 — accuracy-hardened, industry-grade)

Single hand:
  index only ................ move cursor (1-euro smoothing, frozen on pinch)
   + thumb-index pinch ...... left click  /  hold+move = drag
  index+middle, apart ....... scroll (vertical AND horizontal)
  index+middle, touching .... right click
  three fingers (i+m+r) ..... volume (move hand up / down)
  open palm, swipe sideways . switch macOS Space / desktop (Ctrl + arrow)
  fist ...................... neutral (pause)
  rock / pinky .............. configurable macros (see config.json)

Two hands:
  both pinching, move apart/together ... zoom in / out

Advanced infrastructure:
  * threaded capture pipeline (camera on its own thread) + live metrics
    (FPS, inference ms, loop ms, frame-time jitter) on the HUD
  * custom gesture training (ML): press 't', hold a pose, press 1-9 to add
    samples; a KNN classifier recognizes your gesture and fires the hotkey
    bound to it in config.json["gesture_actions"] (only in the 'other' pose,
    so it never fights the built-in controls). Samples persist in gestures.json
  * config.json profile (cutoff, beta, calibration box, macros, gesture table)
  * calibration: press 'c', trace your reach, press 'c' again
  * live tuning keys: a/z = min_cutoff, s/x = beta, [/] = ML match threshold

Accuracy hardening (Phase 4):
  * all gesture thresholds are RATIOS of the hand's own scale, not pixels —
    identical behaviour near or far from the camera
  * pinch uses a smoothed ratio + Schmitt-trigger hysteresis: no boundary
    flicker, no phantom double clicks
  * rotation-invariant finger-extension detection (works with a tilted hand)
  * sub-pixel landmark coordinates (no integer quantization of the cursor)
  * real monotonic timestamps into MediaPipe VIDEO mode (better tracking)
  * per-hand confidence gating; fractional scroll accumulation; structured
    logging; CLI flags; graceful FAILSAFE handling; regression test suite
    (test_virtual_mouse.py)

Quit with 'q', or slam the real cursor into a screen corner (FAILSAFE).
"""

import argparse
import json
import logging
import math
import os
import subprocess
import threading
import time
import socket
import queue
from collections import deque

import cv2
import numpy as np
import pyautogui

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from actions import ActionBus
from app_profiles import AppWatcher, resolve_profile
from gesture_ml import GestureClassifier, landmark_features
from neon_hud import NeonHUD
from pie_menu import PieMenu
from threaded_capture import ThreadedCamera, Metrics

log = logging.getLogger("deepgesture")


# --------------------------------------------------------------------------- #
# Tunable constants
# --------------------------------------------------------------------------- #
CAM_INDEX = 0
HERE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(HERE, "hand_landmarker.task")
CONFIG_PATH = os.path.join(HERE, "config.json")
GESTURE_PATH = os.path.join(HERE, "gestures.json")
FRAME_W, FRAME_H = 1280, 720
CAM_FPS = 60
MARGIN_FRAC = 0.0

EURO_MIN_CUTOFF = 1.2
EURO_BETA = 0.03
EURO_DCUTOFF = 1.0

MAX_JUMP_PX = 160
STABLE_FRAMES = 5


# Geometry thresholds are RATIOS of the hand's own scale (wrist -> middle-MCP
# distance), so behaviour is identical whether the hand is near to or far from
# the camera. The pinch is a Schmitt trigger: it must close below PINCH_ON to
# engage and re-open above PINCH_OFF to release — the dead band kills flicker
# at the boundary (no phantom double clicks).
PINCH_ON = 0.40
PINCH_OFF = 0.55
PINCH_ALPHA = 0.5              # EMA smoothing factor for the pinch ratio
RCLICK_TOUCH_RATIO = 0.35      # index–middle tips "touching" (right click)
FINGER_EXT_RATIO = 1.08        # tip this much farther from wrist than PIP = up
MIN_HAND_SCORE = 0.55          # drop hands MediaPipe is not confident about
RCLICK_DEBOUNCE = 0.6
POSE_COOLDOWN = 0.8

SCROLL_SENSITIVITY = 0.5
SCROLL_DEADZONE = 5
VOL_STEP_FRAC = 0.22           # accumulated hand travel per step, in hand-scale
                               # units (scale-invariant; ~22 px at arm's length)
VOLUME_DEADZONE_PX = 2.0       # per-frame jitter below this doesn't accumulate
VOLUME_DELTA = 6               # percent per step
VOLUME_COOLDOWN = 0.08         # max step rate (the worker coalesces anyway)

ZOOM_STEP_PX = 45
ZOOM_COOLDOWN = 0.25

MENU_HOLD_SEC = 0.8            # fist held this long opens the radial menu
MENU_CLOSE_HOLD = 0.5          # fist again (after leaving fist) closes it

# Open-palm swipe -> switch macOS Space (desktop). A swipe is the palm centre
# travelling > SWIPE_DIST_FRAC of the frame width within SWIPE_WINDOW seconds
# (fractions of frame width -> resolution-independent).
SWIPE_WINDOW = 0.45
SWIPE_DIST_FRAC = 0.22
SWIPE_VERTICAL_FRAC = 0.11     # reject mostly-vertical motion
SWIPE_COOLDOWN = 0.8
WRIST = 0
PALM_CENTER = 9                # middle-finger MCP ~ palm centre

THUMB_TIP = 4
INDEX_TIP, INDEX_PIP = 8, 6
MIDDLE_TIP, MIDDLE_PIP = 12, 10
RING_TIP, RING_PIP = 16, 14
PINKY_TIP, PINKY_PIP = 20, 18

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)

BUILTIN_CONTINUOUS = {"index", "two", "three"}

DEFAULT_CONFIG = {
    "min_cutoff": EURO_MIN_CUTOFF,
    "beta": EURO_BETA,
    "calibration": None,            # {"x0","y0","x1","y1"} in frame px, or null
    "macros": {                     # pose -> hotkey passed to pyautogui.hotkey
        "rock": ["ctrl", "up"],         # Mission Control
        "pinky": ["command", "shift", "3"],  # Screenshot
        # note: "palm" is reserved for the swipe gesture (switch Space)
    },
    "gesture_threshold": 0.85,      # KNN match distance cutoff
    "gesture_actions": {            # trained class -> hotkey (fired in 'other' pose)
        "g1": ["command", "c"],         # default: copy
        "g2": ["command", "v"],         # default: paste
    },
    "swipe": {                      # open-palm swipe bindings (per-app overridable)
        "next": ["ctrl", "right"],
        "prev": ["ctrl", "left"],
    },
    "menu": [                       # radial pie menu (hold a fist to open)
        {"label": "Copy", "keys": ["command", "c"]},
        {"label": "Paste", "keys": ["command", "v"]},
        {"label": "Shot", "keys": ["command", "shift", "3"]},
        {"label": "Mission", "keys": ["ctrl", "up"]},
        {"label": "Spotlight", "keys": ["command", "space"]},
        {"label": "Undo", "keys": ["command", "z"]},
    ],
    "app_profiles": {               # frontmost app -> binding overrides
        "Google Chrome": {
            "swipe": {"next": ["command", "shift", "]"],
                      "prev": ["command", "shift", "["]},
        },
    },
}

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.0


# --------------------------------------------------------------------------- #
# Config persistence
# --------------------------------------------------------------------------- #
def load_config(path=CONFIG_PATH):
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy of defaults
    if os.path.exists(path):
        try:
            with open(path) as fh:
                user = json.load(fh)
            cfg.update({k: user[k] for k in user if k in cfg})
            if isinstance(user.get("macros"), dict):
                cfg["macros"] = user["macros"]
        except Exception as exc:
            log.warning("[config] ignoring unreadable %s: %s", path, exc)
    else:
        save_config(cfg, path)
    return cfg


def save_config(cfg, path=CONFIG_PATH):
    try:
        with open(path, "w") as fh:
            json.dump(cfg, fh, indent=2)
    except Exception as exc:
        log.error("[config] could not save: %s", exc)


# --------------------------------------------------------------------------- #
# 1-euro filter
# --------------------------------------------------------------------------- #
class _LowPass:
    def __init__(self):
        self.s = None

    def __call__(self, value, alpha):
        self.s = value if self.s is None else alpha * value + (1 - alpha) * self.s
        return self.s


class OneEuroFilter:
    def __init__(self, freq=60.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x = _LowPass()
        self._dx = _LowPass()
        self._last_t = None
        self._last_x = None

    def _alpha(self, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x, t):
        if self._last_t is not None and t > self._last_t:
            self.freq = 1.0 / (t - self._last_t)
        self._last_t = t
        dx = 0.0 if self._last_x is None else (x - self._last_x) * self.freq
        self._last_x = x
        edx = self._dx(dx, self._alpha(self.d_cutoff))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self._x(x, self._alpha(cutoff))


# --------------------------------------------------------------------------- #
# Hand tracking — returns ALL hands
# --------------------------------------------------------------------------- #
class HandTracker:
    def __init__(self, model_path=MODEL_PATH):
        if not os.path.exists(model_path):
            raise SystemExit(
                f"Model file not found: {model_path}\n"
                "Download it with:\n  curl -fsSL -o hand_landmarker.task "
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                "hand_landmarker/float16/1/hand_landmarker.task"
            )
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._t0 = time.monotonic()
        self._last_ts = -1

    def find(self, frame_bgr, draw=True):
        """Return list of hands; each hand is a list of 21 sub-pixel (x, y)
        floats. Hands below MIN_HAND_SCORE confidence are dropped."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # Real monotonic milliseconds (strictly increasing, as the Tasks API
        # requires) so MediaPipe's internal temporal smoothing sees true frame
        # spacing instead of a fake fixed-step counter.
        ts = int((time.monotonic() - self._t0) * 1000)
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts
        result = self._landmarker.detect_for_video(mp_image, ts)
        if not result.hand_landmarks:
            return []
        h, w = frame_bgr.shape[:2]
        hands = []
        for i, hand in enumerate(result.hand_landmarks):
            score = (result.handedness[i][0].score
                     if result.handedness and result.handedness[i] else 1.0)
            if score < MIN_HAND_SCORE:
                continue
            # keep float precision — no integer quantization of landmarks
            hands.append([(lm.x * w, lm.y * h) for lm in hand])
        if draw:
            for pts in hands:
                for a, b in HAND_CONNECTIONS:
                    cv2.line(frame_bgr, ipt(pts[a]), ipt(pts[b]), (255, 255, 255), 2)
                for p in pts:
                    cv2.circle(frame_bgr, ipt(p), 3, (0, 0, 255), cv2.FILLED)
        return hands

    def close(self):
        self._landmarker.close()


# --------------------------------------------------------------------------- #
# Gesture helpers
# --------------------------------------------------------------------------- #
def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def ipt(p):
    """Round a sub-pixel landmark to an int pixel tuple for cv2 drawing."""
    return (int(round(p[0])), int(round(p[1])))


def hand_scale(lm):
    """Characteristic hand size in px: wrist -> middle-finger MCP distance.
    Every geometric threshold is expressed relative to this, so gestures are
    invariant to how far the hand is from the camera."""
    return max(distance(lm[WRIST], lm[PALM_CENTER]), 1e-6)


def fingers_up(lm):
    """A finger is extended when its tip is farther from the wrist than its
    PIP joint. Distance-based, so it is rotation-invariant — a tilted or
    sideways hand classifies the same as an upright one (the naive
    tip-above-PIP y-test does not)."""
    w = lm[WRIST]
    return tuple(
        distance(lm[tip], w) > distance(lm[pip], w) * FINGER_EXT_RATIO
        for tip, pip in ((INDEX_TIP, INDEX_PIP), (MIDDLE_TIP, MIDDLE_PIP),
                         (RING_TIP, RING_PIP), (PINKY_TIP, PINKY_PIP))
    )


def classify_pose(lm):
    i, m, r, p = fingers_up(lm)
    if not (i or m or r or p):
        return "fist"
    if i and m and r and p:
        return "palm"
    if i and not m and not r and not p:
        return "index"
    if i and m and not r and not p:
        return "two"
    if i and m and r and not p:
        return "three"
    if i and not m and not r and p:
        return "rock"
    if p and not i and not m and not r:
        return "pinky"
    return "other"


def pinch_ratio(lm):
    """Thumb–index gap as a fraction of the hand's own scale."""
    return distance(lm[THUMB_TIP], lm[INDEX_TIP]) / hand_scale(lm)


def is_pinching(lm):
    return pinch_ratio(lm) < PINCH_ON


class VolumeControl:
    """Async macOS volume. osascript takes 50-150 ms per call — far too slow
    for the frame loop — so all of it runs on a lazy worker thread: the level
    is queried once, then tracked locally, and rapid steps coalesce to the
    newest target (intermediate set calls are skipped, never queued up)."""

    def __init__(self):
        self.level = None              # last known volume 0-100 (None = unknown)
        self._target = None            # newest level to apply
        self._pending = 0              # deltas received before the first query
        self._cv = threading.Condition()
        self._stopped = False
        self._thread = None

    def adjust(self, delta):
        """Nudge volume by delta percent. Returns the new local level."""
        with self._cv:
            if self._thread is None and not self._stopped:
                self._thread = threading.Thread(target=self._work, daemon=True)
                self._thread.start()
            if self.level is None:     # initial query still in flight
                self._pending += delta
                return None
            self.level = max(0, min(100, self.level + delta))
            self._target = self.level
            self._cv.notify()
            return self.level

    def stop(self):
        with self._cv:
            self._stopped = True
            self._cv.notify()

    def _work(self):
        try:
            cur = int(subprocess.check_output(
                ["osascript", "-e", "output volume of (get volume settings)"],
                timeout=2.0).strip())
        except Exception as exc:
            log.warning("volume query failed (%s); assuming 50%%", exc)
            cur = 50
        with self._cv:
            self.level = max(0, min(100, cur + self._pending))
            if self._pending:
                self._target = self.level
            self._pending = 0
        while True:
            with self._cv:
                while self._target is None and not self._stopped:
                    self._cv.wait()
                if self._stopped:
                    return
                tgt, self._target = self._target, None
            try:
                subprocess.run(
                    ["osascript", "-e", f"set volume output volume {tgt}"],
                    check=False, timeout=2.0)
            except Exception as exc:
                log.debug("volume set failed: %s", exc)


# --------------------------------------------------------------------------- #
# Gesture controller
# --------------------------------------------------------------------------- #
class GestureController:
    def __init__(self, screen_w, screen_h, config, app_watcher=None):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.config = config
        self.macros = config.get("macros", {})
        self.calib = config.get("calibration")  # dict or None

        # single exit point for OS input + toast feed
        self.bus = ActionBus(pyautogui)

        # radial pie menu (hold a fist to open)
        self.menu = PieMenu(config.get("menu", []))
        self._fist_since = None
        self._menu_left_fist = False
        self._menu_fist_since = None

        # per-app profiles
        self.watcher = app_watcher
        self.app = ""
        self.profile = resolve_profile(config, "")

        # ML gesture recognition
        self.gesture_actions = config.get("gesture_actions", {})
        self.classifier = GestureClassifier(
            threshold=config.get("gesture_threshold", 0.85)).load(GESTURE_PATH)
        if self.classifier.samples and not self.classifier.weights:
            print("[virtual_mouse] baking gesture model...")
            self.classifier.train_aggresively()
        self.training = False
        self._last_features = None
        self.custom_pred = ("none", 0.0)
        self._custom_cand = None
        self._custom_n = 0
        self._custom_fired = False

        self.fx = OneEuroFilter(CAM_FPS, config["min_cutoff"], config["beta"], EURO_DCUTOFF)
        self.fy = OneEuroFilter(CAM_FPS, config["min_cutoff"], config["beta"], EURO_DCUTOFF)

        # one-shot debounces live in the ActionBus cooldown table now;
        # what remains here is genuine gesture state
        self.pinching = False
        self.pinch_start = 0.0
        self.dragging = False
        self._pinch_lp = _LowPass()    # EMA over the scale-invariant pinch ratio

        self.prev_scroll = None
        self._scroll_rem_v = 0.0       # fractional scroll-tick remainders
        self._scroll_rem_h = 0.0
        self.prev_vol_y = None
        self._vol_accum = 0.0          # hand travel toward the next volume step
        self.volume = VolumeControl()  # async; worker starts on first adjust
        self.prev_zoom = None
        self._swipe_hist = deque()     # (t, palm_x, palm_y) within SWIPE_WINDOW
        self.last_swipe = 0.0          # swipe window state (not just a debounce)

        self._last_idx = None
        self._pose = "neutral"
        self._cand = None
        self._cand_n = 0
        self.last_pose = None

        # calibration capture state
        self.calibrating = False
        self._cal_box = None  # [x0, y0, x1, y1] accumulator

        # New gesture state
        self._last_index_up = 0.0
        self._fist_holding = False

    # ---- tuning / config --------------------------------------------------- #
    def tune(self, d_cutoff=0.0, d_beta=0.0):
        for f in (self.fx, self.fy):
            f.min_cutoff = max(0.05, round(f.min_cutoff + d_cutoff, 3))
            f.beta = max(0.0, round(f.beta + d_beta, 4))
        return self.fx.min_cutoff, self.fx.beta

    def export_config(self):
        return {
            "min_cutoff": self.fx.min_cutoff,
            "beta": self.fx.beta,
            "calibration": self.calib,
            "macros": self.macros,
            "gesture_threshold": self.classifier.threshold,
            "gesture_actions": self.gesture_actions,
            "swipe": self.config.get("swipe", DEFAULT_CONFIG["swipe"]),
            "menu": self.config.get("menu", DEFAULT_CONFIG["menu"]),
            "app_profiles": self.config.get("app_profiles", {}),
        }

    # ---- per-app profiles --------------------------------------------------- #
    def refresh_profile(self, now):
        """Cheap per-frame check; re-resolve bindings only when the app changes."""
        app = self.watcher.current if self.watcher else ""
        if app != self.app:
            self.app = app
            self.profile = resolve_profile(self.config, app)
            self.bus.note("PROFILE", (app or "default").upper()[:22], now=now)

    # ---- ML gesture training / recognition --------------------------------- #
    def observe(self, lm):
        """Compute features + custom prediction without taking any action."""
        self._last_features = landmark_features(lm)
        self.custom_pred = self.classifier.predict(self._last_features)
        return self.custom_pred

    def capture_sample(self, n):
        if self._last_features is not None:
            return self.classifier.add(f"g{n}", self._last_features)
        return 0

    def tune_threshold(self, delta):
        self.classifier.threshold = max(0.1, round(self.classifier.threshold + delta, 3))
        return self.classifier.threshold

    def _custom(self, now):
        label, dist = self.custom_pred
        if label == self._custom_cand:
            self._custom_n += 1
        else:
            self._custom_cand, self._custom_n, self._custom_fired = label, 1, False
        if (self._custom_n >= STABLE_FRAMES and not self._custom_fired
                and label not in ("none", "unknown")):
            keys = self.profile["gesture_actions"].get(label)
            if keys and self.bus.hotkey(keys, key=f"ml:{label}",
                                        cooldown=POSE_COOLDOWN,
                                        detail=label, now=now):
                self._custom_fired = True
                return f"{label} -> {'+'.join(keys)}"
        return f"ML:{label} ({dist:.2f})"

    # ---- radial pie menu ---------------------------------------------------- #
    def _menu(self, lm, now):
        """All gesture dispatch is suspended while the menu is open: point at
        a slice, pinch to fire it, fist again (or timeout) to close."""
        menu = self.menu

        # selection flash in progress -> close once it has played out
        if menu.flash is not None:
            if menu.flash_done(now):
                menu.close()
            return "menu"

        # close: fist held again after having left fist
        raw = classify_pose(lm)
        if raw != "fist":
            self._menu_left_fist = True
            self._menu_fist_since = None
        elif self._menu_left_fist:
            if self._menu_fist_since is None:
                self._menu_fist_since = now
            elif now - self._menu_fist_since > MENU_CLOSE_HOLD:
                menu.close()
                return "menu closed"

        # pinch = select. Once the pinch starts closing (below PINCH_OFF) the
        # hover freezes, because pinching physically drags the index tip —
        # you select what you were pointing at before your fingers moved.
        ratio = self._pinch_lp(pinch_ratio(lm), PINCH_ALPHA)
        hover = menu.update(lm[INDEX_TIP], now, freeze=ratio < PINCH_OFF)
        if not menu.open:                       # timed out
            return "menu timeout"

        sel = menu.try_select(ratio < PINCH_ON, now)
        if sel:
            self.bus.hotkey(sel["keys"], label=f"MENU: {sel['label'].upper()}",
                            key="menu", cooldown=0.3, now=now)
            return f"menu -> {sel['label']}"
        if hover >= 0:
            return f"menu: {menu.slices[hover]['label']}"
        return "menu (point + pinch)"

    # ---- calibration ------------------------------------------------------- #
    def toggle_calibration(self):
        if not self.calibrating:
            self.calibrating = True
            self._cal_box = None
        else:
            self.calibrating = False
            if self._cal_box and (self._cal_box[2] - self._cal_box[0] > 40
                                  and self._cal_box[3] - self._cal_box[1] > 40):
                self.calib = {"x0": self._cal_box[0], "y0": self._cal_box[1],
                              "x1": self._cal_box[2], "y1": self._cal_box[3]}
        return self.calibrating

    def feed_calibration(self, idx):
        x, y = ipt(idx)
        if self._cal_box is None:
            self._cal_box = [x, y, x, y]
        else:
            self._cal_box[0] = min(self._cal_box[0], x)
            self._cal_box[1] = min(self._cal_box[1], y)
            self._cal_box[2] = max(self._cal_box[2], x)
            self._cal_box[3] = max(self._cal_box[3], y)

    def active_region(self, w, h):
        if self.calib:
            return (int(self.calib["x0"]), int(self.calib["y0"]),
                    int(self.calib["x1"]), int(self.calib["y1"]))
        mx, my = int(w * MARGIN_FRAC), int(h * MARGIN_FRAC)
        return mx, my, w - mx, h - my

    # ---- single-hand machinery -------------------------------------------- #
    def _stable_pose(self, lm):
        raw = classify_pose(lm)
        if raw == self._cand:
            self._cand_n += 1
        else:
            self._cand, self._cand_n = raw, 1
        if self._cand_n >= STABLE_FRAMES:
            self._pose = self._cand
        return self._pose

    def _clamp_outlier(self, idx):
        if self._last_idx is not None and distance(idx, self._last_idx) > MAX_JUMP_PX:
            idx = self._last_idx
        self._last_idx = idx
        return idx

    def _move(self, lm, region, now, frame):
        x0, y0, x1, y1 = region
        # Use midpoint of index and middle for smoother 2-finger movement
        idx = (np.array(lm[INDEX_TIP]) + np.array(lm[MIDDLE_TIP])) / 2.0
        idx = self._clamp_outlier(idx.tolist())
        tx = np.interp(idx[0], (x0, x1), (0, self.screen_w))
        ty = np.interp(idx[1], (y0, y1), (0, self.screen_h))
        cx, cy = self.fx(tx, now), self.fy(ty, now)

        self.bus.move(cx, cy)
        cv2.circle(frame, ipt(lm[INDEX_TIP]), 13, (255, 0, 255), 2)
        cv2.circle(frame, ipt(lm[MIDDLE_TIP]), 13, (255, 0, 255), 2)
        return "move"

    def _tap_click(self, now):
        if self.last_pose != "index":
            self.bus.click(now=now)
        return "left click"

    def _hold(self, now):
        if not self._fist_holding:
            self.bus.mouse_down(now)
            self._fist_holding = True
        return "hold (mouse down)"

    def _palm_handler(self, lm, now, frame):
        # Check for swipe first
        res = self._swipe(lm, now, frame)
        if "swipe ->" in res or "swipe <-" in res:
            return res
        
        # If not swiping, treat as right click on entry
        if self.last_pose != "palm":
            self.bus.rclick(now=now)
            return "right click"
        return "palm"

    def _scroll_or_rclick(self, lm, now, frame):
        idx, mid = lm[INDEX_TIP], lm[MIDDLE_TIP]
        if distance(idx, mid) < RCLICK_TOUCH_RATIO * hand_scale(lm):
            self.bus.rclick(now=now, cooldown=RCLICK_DEBOUNCE)
            cv2.circle(frame, ipt(idx), 14, (0, 140, 255), cv2.FILLED)
            return "right click"
        mx_, my_ = (idx[0] + mid[0]) / 2.0, (idx[1] + mid[1]) / 2.0
        if self.prev_scroll is not None:
            dx = mx_ - self.prev_scroll[0]
            dy = self.prev_scroll[1] - my_
            # accumulate fractional ticks so slow, precise scrolling is not
            # truncated to zero; the deadzone still rejects jitter
            if abs(dy) > SCROLL_DEADZONE:
                self._scroll_rem_v += dy * SCROLL_SENSITIVITY
            if abs(dx) > SCROLL_DEADZONE:
                self._scroll_rem_h += dx * SCROLL_SENSITIVITY
            tv, th = int(self._scroll_rem_v), int(self._scroll_rem_h)
            if tv:
                self.bus.scroll(tv, now=now)
                self._scroll_rem_v -= tv
            if th:
                self.bus.hscroll(th, now=now)
                self._scroll_rem_h -= th
        self.prev_scroll = (mx_, my_)
        cv2.circle(frame, ipt((mx_, my_)), 12, (0, 255, 255), 2)
        return "scroll"

    def _volume(self, lm, now):
        """Three fingers: raise/lower the hand to adjust volume. Vertical
        travel ACCUMULATES (scale-invariant) — a smooth slow motion works just
        as well as a flick — and each VOL_STEP_FRAC of hand-scale travel fires
        one +/-VOLUME_DELTA step through the async VolumeControl."""
        my_ = (lm[INDEX_TIP][1] + lm[MIDDLE_TIP][1] + lm[RING_TIP][1]) / 3.0
        if self.prev_vol_y is not None:
            dy = self.prev_vol_y - my_                 # up = positive
            if abs(dy) > VOLUME_DEADZONE_PX:           # ignore standstill jitter
                self._vol_accum += dy / hand_scale(lm)
        self.prev_vol_y = my_

        if abs(self._vol_accum) >= VOL_STEP_FRAC:
            delta = VOLUME_DELTA if self._vol_accum > 0 else -VOLUME_DELTA
            if self.bus.fire("volume", "VOL",
                             lambda: self.volume.adjust(delta),
                             cooldown=VOLUME_COOLDOWN, kind="key",
                             now=now, agg_delta=delta):
                self._vol_accum -= math.copysign(VOL_STEP_FRAC,
                                                 self._vol_accum)
        lvl = self.volume.level
        return f"volume {lvl}%" if lvl is not None else "volume"

    def _swipe(self, lm, now, frame):
        """Open-palm horizontal swipe -> Ctrl+Left/Right (switch Space)."""
        cx, cy = lm[PALM_CENTER]
        fw = frame.shape[1]
        self._swipe_hist.append((now, cx, cy))
        while self._swipe_hist and now - self._swipe_hist[0][0] > SWIPE_WINDOW:
            self._swipe_hist.popleft()

        px, py = ipt((cx, cy))
        cv2.arrowedLine(frame, (px - 60, py), (px + 60, py), (255, 200, 0), 2, tipLength=0.3)
        if now - self.last_swipe < SWIPE_COOLDOWN:
            return "swipe (cooldown)"
        if len(self._swipe_hist) >= 3:
            _, x0, y0 = self._swipe_hist[0]
            dx = cx - x0
            dy = max(abs(p[2] - y0) for p in self._swipe_hist)
            if abs(dx) > SWIPE_DIST_FRAC * fw and dy < SWIPE_VERTICAL_FRAC * fw:
                # mirrored frame: swipe right => "next"; keys come from the
                # active per-app profile (Spaces by default, tabs in Chrome…)
                keys = self.profile["swipe"]["next" if dx > 0 else "prev"]
                self.bus.hotkey(keys, label="SWIPE >" if dx > 0 else "< SWIPE",
                                key="swipe", now=now)
                self.last_swipe = now
                self._swipe_hist.clear()
                return "swipe -> next" if dx > 0 else "swipe <- prev"
        return "swipe ready"

    def _fire_macro(self, pose, now):
        keys = self.profile["macros"].get(pose)
        if keys and self.last_pose != pose:
            self.bus.hotkey(keys, key=f"macro:{pose}", cooldown=POSE_COOLDOWN,
                            detail=pose, now=now)
        return "+".join(keys) if keys else "macro?"

    def process(self, lm, region, now, frame):
        pose = self._stable_pose(lm)
        self.observe(lm)               # features + ML prediction every frame

        # radial menu suspends all other dispatch while open
        if self.menu.open:
            self.last_pose = pose
            return "menu", self._menu(lm, now)

        # Cleanup states when leaving poses
        if pose != "two":
            self.prev_scroll = None
            self._scroll_rem_v = self._scroll_rem_h = 0.0
        if pose != "three":
            self.prev_vol_y = None
            self._vol_accum = 0.0
        if pose != "palm":
            self._swipe_hist.clear()
        if pose != "index":
            self._pinch_lp.s = None
        if pose != "fist":
            if self._fist_holding:
                self.bus.mouse_up(now)
                self._fist_holding = False
            self._fist_since = None

        # Dispatch
        if pose == "two":
            gesture = self._move(lm, region, now, frame)
        elif pose == "index":
            gesture = self._tap_click(now)
        elif pose == "palm":
            gesture = self._palm_handler(lm, now, frame)
        elif pose == "fist":
            gesture = self._hold(now)
        elif pose == "three":
            gesture = self._volume(lm, now)
        elif pose in self.profile["macros"]:
            gesture = self._fire_macro(pose, now)
        else:
            self._last_idx = None
            gesture = self._custom(now)

        self.last_pose = pose
        return pose, gesture

    # ---- two-hand machinery ----------------------------------------------- #
    def process_two_hands(self, hands, now, frame):
        self.menu.close()              # menu is single-hand only
        if self._fist_holding:
            self.bus.mouse_up(now)
            self._fist_holding = False
        h1, h2 = hands[0], hands[1]
        if is_pinching(h1) and is_pinching(h2):
            a, b = h1[INDEX_TIP], h2[INDEX_TIP]
            cv2.line(frame, ipt(a), ipt(b), (0, 255, 0), 2)
            dist = distance(a, b)
            if self.prev_zoom is not None:
                dd = dist - self.prev_zoom
                if dd > ZOOM_STEP_PX:
                    self.bus.hotkey(["command", "="], label="ZOOM +",
                                    key="zoom", cooldown=ZOOM_COOLDOWN, now=now)
                elif dd < -ZOOM_STEP_PX:
                    self.bus.hotkey(["command", "-"], label="ZOOM -",
                                    key="zoom", cooldown=ZOOM_COOLDOWN, now=now)
            self.prev_zoom = dist
            return "2 hands", "zoom"
        self.prev_zoom = None
        return "2 hands", "idle (pinch both to zoom)"


# --------------------------------------------------------------------------- #
# HUD
# --------------------------------------------------------------------------- #
LEGEND = [
    "index        move",
    "+pinch       click / drag",
    "i+m apart    scroll",
    "i+m touch    right click",
    "3 fingers    volume",
    "2h pinch     zoom",
    "palm swipe   switch space",
    "rock/pinky   macros",
    "other pose   ML gesture",
    "fist         pause",
]


def draw_hud(frame, metrics, pose, gesture, region, ctrl):
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = region
    box_color = (0, 255, 255) if ctrl.calib else (0, 180, 255)
    cv2.rectangle(frame, (x0, y0), (x1, y1), box_color, 2)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 62), (30, 30, 30), cv2.FILLED)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, f"FPS {metrics['fps']:4.1f}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 120), 2)
    cv2.putText(frame, f"pose:{pose}", (150, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"action:{gesture}", (340, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    # metrics line
    cv2.putText(frame,
                f"infer {metrics['infer']:4.1f}ms  loop {metrics['loop']:4.1f}ms"
                f"  jitter {metrics['jitter']:4.1f}ms", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 1)
    cv2.putText(frame,
                f"cut {ctrl.fx.min_cutoff:.2f} beta {ctrl.fx.beta:.3f}"
                f"  thr {ctrl.classifier.threshold:.2f}", (640, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 1)

    panel = frame.copy()
    cv2.rectangle(panel, (5, h - 225), (250, h - 5), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(panel, 0.55, frame, 0.45, 0, frame)
    for i, line in enumerate(LEGEND):
        cv2.putText(frame, line, (12, h - 204 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
    cv2.putText(frame, "a/z cut  s/x beta  [/] thr  c calib  t train  q quit",
                (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 200, 255), 1)

    if ctrl.calibrating:
        cv2.putText(frame, "CALIBRATING - trace your reach, press 'c' to finish",
                    (w // 2 - 320, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2)
        if ctrl._cal_box:
            b = ctrl._cal_box
            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)

    if ctrl.training:
        op = frame.copy()
        cv2.rectangle(op, (0, 64), (w, 150), (10, 40, 10), cv2.FILLED)
        cv2.addWeighted(op, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, "TRAINING - hold a pose, press 1-9 to add a sample; "
                    "t to finish", (12, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (120, 255, 120), 2)
        counts = ctrl.classifier.counts()
        txt = "  ".join(f"{k}:{v}" for k, v in sorted(counts.items())) or "(no samples yet)"
        cv2.putText(frame, txt, (12, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (200, 255, 200), 2)


# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Hand-gesture virtual mouse (webcam -> cursor/gestures)")
    ap.add_argument("--camera", type=int, default=CAM_INDEX, help="camera index")
    ap.add_argument("--width", type=int, default=FRAME_W, help="capture width")
    ap.add_argument("--height", type=int, default=FRAME_H, help="capture height")
    ap.add_argument("--fps", type=int, default=CAM_FPS, help="capture FPS")
    ap.add_argument("--model", default=MODEL_PATH, help="hand_landmarker.task path")
    ap.add_argument("--hud", choices=("neon", "classic"), default="neon",
                    help="overlay style (default: neon)")
    ap.add_argument("--headless", action="store_true", help="run without window")
    ap.add_argument("--ipc-port", type=int, default=50051, help="IPC port")
    ap.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return ap.parse_args(argv)


class DaemonIPC:
    def __init__(self, port):
        self.port = port
        self.state = {"fps": 0, "pose": "neutral", "gesture": "-", "training": False}
        self.commands = queue.Queue()
        self.show_window = False
        self._stop = False

    def start(self):
        t = threading.Thread(target=self._run, daemon=True)
        t.start()
        return self

    def _run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", self.port))
                s.listen()
            except Exception as e:
                log.error("Could not bind IPC port %d: %s", self.port, e)
                return
            s.settimeout(0.5)
            while not self._stop:
                try:
                    conn, _ = s.accept()
                    with conn:
                        data = conn.recv(1024)
                        if data:
                            req = json.loads(data.decode("utf-8"))
                            cmd = req.get("cmd")
                            if cmd == "GET_STATE":
                                conn.sendall(json.dumps(self.state).encode("utf-8"))
                            elif cmd:
                                self.commands.put(cmd)
                                conn.sendall(b'{"status": "ok"}')
                except socket.timeout:
                    continue
                except Exception:
                    continue


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    config = load_config()
    screen_w, screen_h = pyautogui.size()
    log.info("screen %dx%d, capture %dx%d@%d, camera %d",
             screen_w, screen_h, args.width, args.height, args.fps, args.camera)
    camera = ThreadedCamera(args.camera, args.width, args.height, args.fps).start()
    if not camera.isOpened():
        raise SystemExit(
            "Could not open the webcam. On macOS, grant Camera permission to "
            "your terminal in System Settings > Privacy & Security > Camera."
        )

    tracker = HandTracker(args.model)
    watcher = AppWatcher().start()
    ctrl = GestureController(screen_w, screen_h, config, app_watcher=watcher)
    metrics = Metrics()
    hud = (NeonHUD(PINCH_ON, PINCH_OFF, SWIPE_DIST_FRAC)
           if args.hud == "neon" else None)
    
    current_cam_idx = args.camera

    try:
        while True:
            t0 = time.time()
            ok, frame = camera.read()
            if not ok or frame is None:
                if cv2.waitKey(5) & 0xFF == ord("q"):
                    break
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            now = time.time()
            metrics.tick(now)
            ctrl.refresh_profile(now)

            t_inf = time.time()
            hands = tracker.find(frame, draw=hud is None)
            metrics.add_infer((time.time() - t_inf) * 1000.0)

            region = ctrl.active_region(w, h)
            pose, gesture = ("no hand", "-")

            try:
                if ctrl.training:
                    pose, gesture = "train", "press 1-9 to sample"
                    if hands:
                        ctrl.observe(hands[0])
                elif ctrl.calibrating:
                    pose, gesture = "calib", "tracing reach"
                    if hands:
                        ctrl.feed_calibration(hands[0][INDEX_TIP])
                elif len(hands) >= 2:
                    pose, gesture = ctrl.process_two_hands(hands, now, frame)
                elif len(hands) == 1:
                    pose, gesture = ctrl.process(hands[0], region, now, frame)
            except pyautogui.FailSafeException:
                log.warning("FAILSAFE corner hit — exiting cleanly")
                break

            stats = metrics.stats()
            if hud is not None:
                hud.draw(frame, stats, pose, gesture, region, ctrl, hands, now)
            else:
                draw_hud(frame, stats, pose, gesture, region, ctrl)

            cv2.imshow("Virtual Mouse", frame)
            metrics.add_loop((time.time() - t0) * 1000.0)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("n"): # 'n' for next camera
                log.info("Switching camera...")
                camera.release()
                current_cam_idx = (current_cam_idx + 1) % 4
                camera = ThreadedCamera(current_cam_idx, args.width, args.height, args.fps).start()
            elif key == ord("a"):
                ctrl.tune(d_cutoff=-0.1)
            elif key == ord("z"):
                ctrl.tune(d_cutoff=+0.1)
            elif key == ord("s"):
                ctrl.tune(d_beta=-0.005)
            elif key == ord("x"):
                ctrl.tune(d_beta=+0.005)
            elif key == ord("["):
                ctrl.tune_threshold(-0.05)
            elif key == ord("]"):
                ctrl.tune_threshold(+0.05)
            elif key == ord("c") and not ctrl.training:
                on = ctrl.toggle_calibration()
                if not on:
                    save_config(ctrl.export_config())
            elif key == ord("t") and not ctrl.calibrating:
                ctrl.training = not ctrl.training
                if not ctrl.training:
                    print("[virtual_mouse] training model aggresively...")
                    ctrl.classifier.train_aggresively()
                    ctrl.classifier.save(GESTURE_PATH)
                    save_config(ctrl.export_config())
            elif ctrl.training and ord("1") <= key <= ord("9"):
                ctrl.capture_sample(int(chr(key)))
    finally:
        if ctrl.dragging:
            ctrl.bus.mouse_up(quiet=True)
        ctrl.classifier.save(GESTURE_PATH)
        save_config(ctrl.export_config())
        ctrl.volume.stop()
        watcher.stop()
        tracker.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
