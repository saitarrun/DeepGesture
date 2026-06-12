"""
Neon HUD — a sci-fi, gesture-reactive overlay for Deep Gesture.

Everything on screen is driven by the same state the gesture engine uses, so
the UI *is* the feedback loop:

  * glowing hand skeleton, tinted by the active pose's color
  * rotating reticle on the index fingertip + a pinch-charge ring that fills
    as thumb and index close (full ring = click threshold; pulses while
    pinched, amber while dragging)
  * fading motion trail behind the cursor finger
  * swipe energy bar with marching chevrons under an open palm — fills toward
    the trigger distance in the direction you are sweeping
  * animated corner brackets + marching dashes around the active region
  * top status bar with a center pose notch, FPS sparkline, pipeline metrics
  * gesture map panel that highlights the live pose row
  * ML readout (label + confidence bar) when a custom gesture is in play
  * pulsing REC overlay in training mode, animated crosshair in calibration

The glow is real: neon strokes are drawn to a separate layer, blurred at
quarter resolution, and added back — cheap enough to run every frame.

Cyberpunk layer (Phase 6): a CRT ambience pass grades the camera image
(duotone LUT + scanlines + vignette + roaming scan sweep), the pose title gets
chromatic aberration, every action firing on the bus triggers a ~120 ms glitch
burst (shifted row bands + RGB split), and startup plays a boot sequence.
All motion is eased: a frame-rate-independent exponential `_ease` drives the
accent-color crossfade between poses, pinch-gauge and swipe-charge fills,
pie-menu hover heat (slices warm up / cool down), the legend highlight bar
(slides between rows), and cubic ease-out toast slide-ins.

Pure OpenCV + NumPy; no extra dependencies. Render with:

    NeonHUD(pinch_on, pinch_off, swipe_frac).draw(frame, stats, pose, gesture,
                                                  region, ctrl, hands, now)
"""

import math
from collections import deque

import cv2
import numpy as np

from pie_menu import INNER_R, OUTER_R, OPEN_ANIM

FONT = cv2.FONT_HERSHEY_SIMPLEX

# landmark indices (MediaPipe 21-point hand model)
THUMB_TIP, INDEX_TIP, MIDDLE_TIP, PALM_CENTER, WRIST = 4, 8, 12, 9, 0

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)

POSE_COLORS = {                      # BGR accents, one per mode
    "index": (255, 220, 0),          # cyan        — cursor
    "two": (0, 230, 255),            # yellow      — scroll
    "three": (255, 0, 230),          # magenta     — volume
    "palm": (0, 150, 255),           # orange      — swipe
    "rock": (180, 60, 255),          # pink        — macro
    "pinky": (255, 130, 90),         # periwinkle  — macro
    "fist": (140, 140, 140),         # gray        — pause
    "other": (90, 255, 120),         # green       — ML gesture
    "2 hands": (255, 90, 200),       # violet      — zoom
    "menu": (200, 90, 255),          # pink        — radial menu
}

KIND_COLORS = {                      # toast accent by event kind
    "mouse": (255, 220, 0),          # cyan
    "key": (0, 150, 255),            # orange
    "info": (200, 90, 255),          # pink (profile switches etc.)
    "action": (200, 200, 200),
}

TOAST_TTL = 2.5                      # seconds a toast stays on screen

LEGEND_ROWS = (                      # ASCII only: Hershey fonts can't render ·/—
    ("index", "INDEX", "move, pinch=click, drag"),
    ("two", "2 FINGERS", "scroll, touch=r-click"),
    ("three", "3 FINGERS", "volume up / down"),
    ("palm", "PALM", "swipe = switch space"),
    ("rock", "ROCK", "macro"),
    ("pinky", "PINKY", "macro"),
    ("other", "CUSTOM", "ML gesture"),
    ("fist", "FIST", "pause, hold = menu"),
    ("2 hands", "2 HANDS", "pinch both = zoom"),
)


def _ip(p):
    return (int(round(p[0])), int(round(p[1])))


BOOT_SEC = 1.6                       # startup boot-sequence duration


class NeonHUD:
    def __init__(self, pinch_on, pinch_off, swipe_frac):
        self.pinch_on = pinch_on
        self.pinch_off = pinch_off
        self.swipe_frac = swipe_frac
        self.trail = deque(maxlen=24)
        self.fps_hist = deque(maxlen=120)

        # animation state — everything on screen eases instead of snapping
        self._t_prev = None            # last draw time (for dt)
        self._dt = 0.0
        self._accent = None            # smoothed accent color (floats)
        self._gauge = None             # eased pinch-gauge fill
        self._charge = 0.0             # eased swipe-charge fill
        self._vol_fill = None          # eased volume-meter fill
        self._hover = {}               # menu slice idx -> hover heat 0..1
        self._legend_row = None        # eased legend highlight position
        self._boot_at = None           # first-draw time (boot sequence)
        self._glitch_until = 0.0       # glitch burst deadline
        self._last_event_t = None      # newest bus event seen (glitch trigger)
        self._last_pose = None

    # ------------------------------------------------------------------ #
    # easing
    # ------------------------------------------------------------------ #
    @staticmethod
    def _ease(cur, target, dt, speed):
        """Frame-rate independent exponential approach (smooth, no overshoot)."""
        return cur + (target - cur) * (1.0 - math.exp(-speed * dt))

    @staticmethod
    def _ease_out(p):
        p = max(0.0, min(1.0, p))
        return 1.0 - (1.0 - p) ** 3

    # ------------------------------------------------------------------ #
    def draw(self, frame, stats, pose, gesture, region, ctrl, hands, now):
        h, w = frame.shape[:2]
        first = self._t_prev is None
        self._dt = 0.0 if first else min(max(now - self._t_prev, 0.0), 0.05)
        self._t_prev = now
        if self._boot_at is None:
            self._boot_at = now
        self.fps_hist.append(stats["fps"])

        # accent crossfades between poses instead of snapping
        target = POSE_COLORS.get(pose, (180, 180, 180))
        if self._accent is None:
            self._accent = [float(c) for c in target]
        else:
            self._accent = [self._ease(c, t, self._dt, 10.0)
                            for c, t in zip(self._accent, target)]
        accent = tuple(int(c) for c in self._accent)

        # ---------- glow layer: neon strokes, blurred + added back ----------
        glow = np.zeros_like(frame)
        self._region_brackets(glow, region, accent, now)
        for lm in hands:
            self._skeleton(glow, lm, accent)
        menu = getattr(ctrl, "menu", None)
        if menu is not None and menu.open:
            self._pie_menu_glow(glow, menu, accent, now)
        busy = ctrl.training or ctrl.calibrating or (menu is not None and menu.open)
        if hands and not busy:
            lm = hands[0]
            if pose == "index":
                self.trail.append(lm[INDEX_TIP])
                self._trail(glow, accent)
                self._reticle(glow, lm[INDEX_TIP], accent, now)
                self._pinch_gauge(glow, lm, ctrl, now)
            elif pose == "two":
                mid = ((lm[INDEX_TIP][0] + lm[MIDDLE_TIP][0]) / 2,
                       (lm[INDEX_TIP][1] + lm[MIDDLE_TIP][1]) / 2)
                self._reticle(glow, mid, accent, now, r=20)
                self.trail.clear()
            elif pose == "three":
                self._volume_meter(glow, lm, ctrl, accent, now)
                self.trail.clear()
            elif pose == "palm":
                self._swipe_charge(glow, lm, ctrl, w, now)
                self.trail.clear()
            else:
                self.trail.clear()
        else:
            self.trail.clear()
        self._composite_glow(frame, glow)

        # crisp landmark dots over the glow
        for lm in hands:
            for p in lm:
                cv2.circle(frame, _ip(p), 2, (255, 255, 255), -1)

        # ---------- panels & text (crisp) ----------
        self._top_bar(frame, stats, pose, gesture, ctrl, accent, w)
        self._legend(frame, pose, accent, h)
        self._toasts(frame, ctrl, now, w)
        if menu is not None and menu.open:
            self._pie_menu_labels(frame, menu, accent, now)
        if pose == "other" and not busy:
            self._ml_readout(frame, ctrl, w, h)
        if ctrl.calibrating:
            self._calibration(frame, ctrl, now, w, h)
        if ctrl.training:
            self._training(frame, ctrl, now, w, h)

        if menu is None or not menu.open:
            self._hover.clear()                      # reset hover heat
        self._boot(frame, now, accent, w, h)

    # ------------------------------------------------------------------ #
    # boot sequence
    # ------------------------------------------------------------------ #
    def _boot(self, frame, now, accent, w, h):
        """Startup: a reveal line sweeps across; text types itself out."""
        t = now - self._boot_at
        if t >= BOOT_SEC:
            return
        e = self._ease_out(t / BOOT_SEC)
        x = int(w * e)
        if x < w:
            # Clean darkening of unrevealed area
            roi = frame[:, x:]
            cv2.multiply(roi, (0.3, 0.3, 0.3, 1.0), roi)
            cv2.line(frame, (x, 0), (x, h), accent, 2)
        msg = "SYSTEM READY // V-MOUSE CYBERDECK ONLINE"
        n = int(len(msg) * min(1.0, t / (BOOT_SEC * 0.7)))
        cursor = "_" if int(now * 3) % 2 else " "
        cv2.putText(frame, msg[:n] + cursor, (w // 2 - 250, h - 60), FONT,
                    0.65, (230, 255, 255), 2)

    # ------------------------------------------------------------------ #
    # glow-layer elements
    # ------------------------------------------------------------------ #
    @staticmethod
    def _composite_glow(frame, glow):
        h, w = frame.shape[:2]
        small = cv2.resize(glow, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 3)
        halo = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        cv2.add(frame, halo, frame)                       # soft halo
        cv2.addWeighted(frame, 1.0, glow, 0.85, 0, frame)  # crisp core

    @staticmethod
    def _skeleton(img, lm, accent):
        dim = tuple(int(c * 0.4) for c in accent)
        bright = tuple(int(c * 0.9) for c in accent)
        for a, b in HAND_CONNECTIONS:
            cv2.line(img, _ip(lm[a]), _ip(lm[b]), dim, 4)
            cv2.line(img, _ip(lm[a]), _ip(lm[b]), bright, 1)

    @staticmethod
    def _region_brackets(img, region, accent, now):
        x0, y0, x1, y1 = (int(v) for v in region)
        L = 26
        for cx, cy, dx, dy in ((x0, y0, 1, 1), (x1, y0, -1, 1),
                               (x0, y1, 1, -1), (x1, y1, -1, -1)):
            cv2.line(img, (cx, cy), (cx + dx * L, cy), accent, 3)
            cv2.line(img, (cx, cy), (cx, cy + dy * L), accent, 3)
        dim = tuple(int(c * 0.45) for c in accent)
        period, dash = 48, 14
        off = int(now * 60) % period
        for x in range(x0 + off, x1 - dash, period):       # marching dashes
            cv2.line(img, (x, y0), (x + dash, y0), dim, 1)
            cv2.line(img, (x, y1), (x + dash, y1), dim, 1)
        for y in range(y0 + off, y1 - dash, period):
            cv2.line(img, (x0, y), (x0, y + dash), dim, 1)
            cv2.line(img, (x1, y), (x1, y + dash), dim, 1)

    @staticmethod
    def _reticle(img, center, accent, now, r=26):
        c = _ip(center)
        r = int(r + 2.5 * math.sin(now * 4.0))             # breathing
        a = (now * 180.0) % 360.0
        # Multi-layered technical reticle
        for off in (0, 120, 240):
            cv2.ellipse(img, c, (r, r), a + off, 0, 45, accent, 2)
        for off in (60, 180, 300):
            cv2.ellipse(img, c, (r + 8, r + 8), -a * 1.5 + off, 0, 30, accent, 1)
        # Static corner markers
        s = r + 14
        for dx, dy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            cv2.line(img, (c[0] + dx * s, c[1] + dy * (s - 6)), (c[0] + dx * s, c[1] + dy * s), accent, 1)
            cv2.line(img, (c[0] + dx * (s - 6), c[1] + dy * s), (c[0] + dx * s, c[1] + dy * s), accent, 1)

    def _pinch_gauge(self, img, lm, ctrl, now):
        """Ring that fills as thumb & index close: full ring = click point."""
        ratio = getattr(ctrl, "_pinch_lp", None) and ctrl._pinch_lp.s
        if ratio is None:
            return
        c = _ip(lm[INDEX_TIP])
        if ctrl.dragging:
            color, p, r = (0, 170, 255), 1.0, 46           # amber: dragging
        elif ctrl.pinching:
            pulse = int(4 * math.sin(now * 15.0))
            color, p, r = (0, 255, 120), 1.0, 44 + pulse   # green: pinched
        else:
            span = max(0.95 - self.pinch_on, 1e-6)
            p = max(0.0, min(1.0, (0.95 - ratio) / span))
            color = (0, 255, 120) if p >= 1.0 else (255, 255, 255)
            r = 44
        self._gauge = p if self._gauge is None else \
            self._ease(self._gauge, p, self._dt, 18.0)     # eased fill
        
        # Outer ring
        cv2.ellipse(img, c, (r, r), -90, 0, int(360 * self._gauge), color, 3)
        # Inner technical detail ring
        if self._gauge > 0.1:
            cv2.ellipse(img, c, (r-6, r-6), 90, 0, int(-360 * self._gauge * 0.7), color, 1)
        
        cv2.line(img, _ip(lm[THUMB_TIP]), c, color, 2)
        if ctrl.dragging:
            cv2.putText(img, "DRAG", (c[0] + 52, c[1] + 5), FONT, 0.55, color, 2)

    def _trail(self, img, accent):
        pts = list(self.trail)
        n = len(pts)
        for i in range(1, n):
            f = i / n                                       # fades toward the tail
            col = tuple(int(c * f) for c in accent)
            cv2.line(img, _ip(pts[i - 1]), _ip(pts[i]), col, max(1, int(3 * f)))

    def _volume_meter(self, img, lm, ctrl, accent, now):
        """Vertical level bar beside the hand while in volume mode."""
        lvl = getattr(getattr(ctrl, "volume", None), "level", None)
        target = (lvl if lvl is not None else 0) / 100.0
        self._vol_fill = target if self._vol_fill is None else \
            self._ease(self._vol_fill, target, self._dt, 12.0)
        cx, cy = _ip(lm[PALM_CENTER])
        x, y0, y1 = cx + 120, cy - 100, cy + 100
        dim = tuple(int(c * 0.3) for c in accent)
        # Holographic base
        cv2.rectangle(img, (x, y0), (x + 18, y1), dim, 1)
        cv2.rectangle(img, (x - 4, y1 + 4), (x + 22, y1 + 8), dim, -1)
        
        fy = int(y1 - (y1 - y0) * self._vol_fill)
        if fy < y1 - 1:
            cv2.rectangle(img, (x + 3, fy), (x + 15, y1 - 2), accent, -1)
            # Active level pulse
            glow_w = int(2 + 2 * math.sin(now * 12))
            cv2.rectangle(img, (x - glow_w, fy - 1), (x + 18 + glow_w, fy + 1), (255, 255, 255), -1)
            
        for k in range(1, 11):                          # 10% tick marks
            ty = y1 - int((y1 - y0) * k / 10)
            w = 8 if k % 5 == 0 else 4
            cv2.line(img, (x - w, ty), (x - 1, ty), dim, 1)
        txt = f"{lvl}%" if lvl is not None else "--"
        cv2.putText(img, txt, (x - 10, y0 - 18), FONT, 0.6, accent, 2)
        cv2.putText(img, "VOL", (x - 6, y1 + 28), FONT, 0.45, dim, 1)

    def _swipe_charge(self, img, lm, ctrl, w, now):
        """Energy bar under the palm filling toward the swipe trigger."""
        cx, cy = _ip(lm[PALM_CENTER])
        hist = getattr(ctrl, "_swipe_hist", ())
        dx = (hist[-1][1] - hist[0][1]) if len(hist) >= 2 else 0.0
        need = max(self.swipe_frac * w, 1e-6)
        self._charge = self._ease(self._charge,
                                  max(-1.0, min(1.0, dx / need)),
                                  self._dt, 14.0)          # eased fill
        p = self._charge
        color = (0, 150, 255)
        bw, y = 150, cy + 95
        cv2.rectangle(img, (cx - bw, y - 7), (cx + bw, y + 7), (90, 90, 90), 1)
        fill = int(bw * p)
        if fill:
            cv2.rectangle(img, (cx, y - 6), (cx + fill, y + 6),
                          (0, 255, 160) if abs(p) >= 1.0 else color, -1)
        cv2.line(img, (cx, y - 10), (cx, y + 10), (200, 200, 200), 1)
        step = int(now * 80) % 16                           # marching chevrons
        for k in range(3):
            o = 30 + k * 16 + step
            for s in (1, -1):
                pts = np.array([(cx + s * o, cy - 12), (cx + s * (o + 10), cy),
                                (cx + s * o, cy + 12)], np.int32)
                cv2.polylines(img, [pts], False, color, 2)

    # ------------------------------------------------------------------ #
    # radial pie menu
    # ------------------------------------------------------------------ #
    @staticmethod
    def _menu_geometry(menu, now):
        """Shared geometry: eased open-animation scale + ring radii."""
        k = min(1.0, (now - menu.opened_at) / OPEN_ANIM)
        k = 1.0 - (1.0 - k) ** 2                       # ease-out bloom
        inner = max(10, int(INNER_R * k))
        outer = max(inner + 12, int(OUTER_R * k))
        return inner, outer

    def _pie_menu_glow(self, glow, menu, accent, now):
        if not menu.slices:
            return
        cx, cy = _ip(menu.anchor)
        inner, outer = self._menu_geometry(menu, now)
        rmid = (inner + outer) // 2
        thick = max(8, (outer - inner) // 2)           # ring, not a solid disc
        span = 360.0 / len(menu.slices)
        for i in range(len(menu.slices)):
            a0 = -90.0 - span / 2.0 + i * span + 8.0   # 8 deg gaps survive the blur
            a1 = a0 + span - 16.0
            flash = menu.flash is not None and menu.flash[0] == i
            hovered = i == menu.hover and menu.flash is None
            # hover heat eases in/out -> slices warm up and cool down smoothly
            heat = self._ease(self._hover.get(i, 0.0),
                              1.0 if hovered else 0.0, self._dt, 14.0)
            self._hover[i] = heat
            if flash:
                col, t = (255, 255, 255), thick + 14
            else:
                col = tuple(int(c * (0.25 + 0.75 * heat)) for c in accent)
                t = thick + int(10 * heat)
            cv2.ellipse(glow, (cx, cy), (rmid, rmid), 0, a0, a1, col, t)
        cv2.circle(glow, (cx, cy), max(4, inner - 8),
                   tuple(int(c * 0.5) for c in accent), 2)

    def _pie_menu_labels(self, frame, menu, accent, now):
        if not menu.slices:
            return
        cx, cy = _ip(menu.anchor)
        inner, outer = self._menu_geometry(menu, now)
        span = 360.0 / len(menu.slices)
        for i, sl in enumerate(menu.slices):
            am = math.radians(-90.0 + i * span)
            heat = self._hover.get(i, 0.0)
            lx = cx + int((outer + 28 + 8 * heat) * math.cos(am))
            ly = cy + int((outer + 28 + 8 * heat) * math.sin(am))
            scale = 0.42 + 0.13 * heat                 # label grows with heat
            text = sl["label"].upper()
            size = cv2.getTextSize(text, FONT, scale, 2)[0]
            g = int(185 + 70 * heat)
            cv2.putText(frame, text, (lx - size[0] // 2, ly + 5), FONT, scale,
                        (g, g, g), 2 if heat > 0.5 else 1)
        cv2.putText(frame, "POINT + PINCH = SELECT   FIST = CLOSE",
                    (cx - 158, cy + outer + 56), FONT, 0.45, (210, 210, 210), 1)

    # ------------------------------------------------------------------ #
    # toast feed (events from the ActionBus)
    # ------------------------------------------------------------------ #
    def _toasts(self, frame, ctrl, now, w):
        bus = getattr(ctrl, "bus", None)
        if bus is None:
            return
        y = 92
        for ev in reversed(bus.events):                # newest on top
            age = now - ev["t"]
            if age > TOAST_TTL or age < 0:
                continue
            fade = 1.0 - age / TOAST_TTL
            slide = int(28 * (1.0 - self._ease_out(age / 0.18)))  # cubic slide-in
            text = ev["label"] + (f"  {ev['detail']}" if ev["detail"] else "")
            size = cv2.getTextSize(text, FONT, 0.5, 1)[0]
            x1 = w - 14 + slide
            x0 = x1 - size[0] - 28
            self._fill(frame, np.array([(x0, y - 18), (x1, y - 18),
                                        (x1, y + 8), (x0, y + 8)], np.int32),
                       alpha=0.18 + 0.42 * fade)
            kc = KIND_COLORS.get(ev["kind"], KIND_COLORS["action"])
            cv2.rectangle(frame, (x0, y - 18), (x0 + 4, y + 8),
                          tuple(int(c * fade) for c in kc), -1)
            g = int(90 + 145 * fade)
            cv2.putText(frame, text, (x0 + 12, y), FONT, 0.5, (g, g, g), 1)
            y += 34
            if y > 92 + 5 * 34:
                break

    # ------------------------------------------------------------------ #
    # crisp panels
    # ------------------------------------------------------------------ #
    @staticmethod
    def _fill(frame, pts, color=(15, 12, 8), alpha=0.35):
        ov = frame.copy()
        cv2.fillPoly(ov, [pts], color)
        cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0, frame)

    def _top_bar(self, frame, stats, pose, gesture, ctrl, accent, w):
        cx = w // 2
        pts = np.array([(0, 0), (w, 0), (w, 44), (cx + 190, 44), (cx + 160, 72),
                        (cx - 160, 72), (cx - 190, 44), (0, 44)], np.int32)
        self._fill(frame, pts)
        cv2.polylines(frame, [pts[2:8]], False, accent, 1)

        label = pose.upper()
        size = cv2.getTextSize(label, FONT, 0.85, 2)[0]
        lx = cx - size[0] // 2
        cv2.putText(frame, label, (lx, 34), FONT, 0.85, accent, 2)
        gsize = cv2.getTextSize(gesture, FONT, 0.45, 1)[0]
        cv2.putText(frame, gesture, (cx - gsize[0] // 2, 62), FONT, 0.45,
                    (235, 235, 235), 1)

        cv2.putText(frame, f"{stats['fps']:5.1f} FPS", (12, 20), FONT, 0.55,
                    (0, 255, 140), 2)
        self._sparkline(frame, 12, 27, 150, 40)

        # active per-app profile chip
        app = (getattr(ctrl, "app", "") or "default").upper()[:20]
        size = cv2.getTextSize(app, FONT, 0.42, 1)[0]
        x0, x1 = 172, 172 + size[0] + 16
        cv2.rectangle(frame, (x0, 8), (x1, 30),
                      tuple(int(c * 0.6) for c in accent), 1)
        cv2.putText(frame, app, (x0 + 8, 24), FONT, 0.42, (210, 210, 210), 1)
        cv2.putText(frame,
                    f"INF {stats['infer']:4.1f}ms  LOOP {stats['loop']:4.1f}ms"
                    f"  JIT {stats['jitter']:4.1f}ms",
                    (w - 388, 18), FONT, 0.45, (180, 220, 255), 1)
        cv2.putText(frame,
                    f"CUT {ctrl.fx.min_cutoff:.2f}  BETA {ctrl.fx.beta:.3f}"
                    f"  THR {ctrl.classifier.threshold:.2f}",
                    (w - 388, 38), FONT, 0.45, (180, 220, 255), 1)

    def _sparkline(self, frame, x0, y0, x1, y1):
        hist = list(self.fps_hist)[-60:]
        if len(hist) < 2:
            return
        top = max(60.0, max(hist))
        n = len(hist)
        pts = np.array(
            [(x0 + int(i * (x1 - x0) / (n - 1)),
              y1 - int(min(v, top) / top * (y1 - y0))) for i, v in enumerate(hist)],
            np.int32)
        cv2.polylines(frame, [pts], False, (0, 255, 140), 1)

    def _legend(self, frame, pose, accent, h):
        x, y0 = 10, h - 232
        self._fill(frame, np.array([(x, y0), (x + 256, y0),
                                    (x + 256, h - 8), (x, h - 8)], np.int32))
        cv2.putText(frame, "GESTURE MAP", (x + 10, y0 + 18), FONT, 0.5,
                    (210, 210, 210), 1)
        cv2.line(frame, (x + 8, y0 + 26), (x + 248, y0 + 26), (90, 90, 90), 1)
        # highlight bar slides smoothly between rows
        keys = [k for k, _, _ in LEGEND_ROWS]
        if pose in keys:
            tgt = float(keys.index(pose))
            self._legend_row = tgt if self._legend_row is None else \
                self._ease(self._legend_row, tgt, self._dt, 14.0)
        if self._legend_row is not None:
            yy = y0 + 44 + int(self._legend_row * 18)
            cv2.rectangle(frame, (x + 4, yy - 12), (x + 252, yy + 4),
                          tuple(int(c * 0.35) for c in accent), -1)
            cv2.rectangle(frame, (x + 4, yy - 12), (x + 7, yy + 4), accent, -1)
        for i, (key, name, act) in enumerate(LEGEND_ROWS):
            yy = y0 + 44 + i * 18
            cur = key == pose
            row_col = POSE_COLORS.get(key, (170, 170, 170))
            cv2.putText(frame, name, (x + 12, yy), FONT, 0.42,
                        row_col if cur else (160, 160, 160), 1)
            cv2.putText(frame, act, (x + 100, yy), FONT, 0.38,
                        (255, 255, 255) if cur else (130, 130, 130), 1)
        cv2.putText(frame, "a/z cut  s/x beta  [/] thr  c calib  t train  q quit",
                    (x + 10, h - 14), FONT, 0.38, (150, 195, 255), 1)

    def _ml_readout(self, frame, ctrl, w, h):
        label, dist = ctrl.custom_pred
        thr = max(ctrl.classifier.threshold, 1e-6)
        conf = max(0.0, min(1.0, 1.0 - dist / thr))
        x, y = w // 2 - 130, h - 64
        self._fill(frame, np.array([(x, y), (x + 260, y),
                                    (x + 260, y + 50), (x, y + 50)], np.int32))
        col = (90, 255, 120) if label not in ("none", "unknown") else (140, 140, 140)
        cv2.putText(frame, f"ML  {label.upper()}", (x + 12, y + 21), FONT, 0.55, col, 2)
        cv2.putText(frame, f"{conf * 100:3.0f}%", (x + 205, y + 21), FONT, 0.5, col, 1)
        cv2.rectangle(frame, (x + 12, y + 32), (x + 248, y + 42), (80, 80, 80), 1)
        cv2.rectangle(frame, (x + 12, y + 32), (x + 12 + int(236 * conf), y + 42),
                      col, -1)

    def _calibration(self, frame, ctrl, now, w, h):
        cx, cy = w // 2, h // 2
        a = (now * 120.0) % 360.0
        cv2.ellipse(frame, (cx, cy), (34, 34), a, 0, 90, (0, 255, 255), 2)
        cv2.ellipse(frame, (cx, cy), (34, 34), a + 180, 0, 90, (0, 255, 255), 2)
        cv2.line(frame, (cx - 14, cy), (cx + 14, cy), (0, 255, 255), 1)
        cv2.line(frame, (cx, cy - 14), (cx, cy + 14), (0, 255, 255), 1)
        cv2.putText(frame, "CALIBRATING - trace your reach, 'c' to lock",
                    (cx - 290, cy - 58), FONT, 0.75, (0, 255, 255), 2)
        if ctrl._cal_box:
            b = [int(v) for v in ctrl._cal_box]
            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 120), 2)

    def _training(self, frame, ctrl, now, w, h):
        pulse = 0.5 + 0.5 * math.sin(now * 6.0)
        cv2.rectangle(frame, (3, 3), (w - 4, h - 4),
                      (0, 0, int(140 + 110 * pulse)), 3)
        y = 96
        self._fill(frame, np.array([(0, y - 26), (w, y - 26),
                                    (w, y + 56), (0, y + 56)], np.int32),
                   color=(10, 30, 10))
        if pulse > 0.4:
            cv2.circle(frame, (24, y - 6), 7, (0, 0, 255), -1)
        cv2.putText(frame, "REC  TRAINING - hold a pose, press 1-9 to sample,"
                    " 't' to finish", (42, y), FONT, 0.6, (120, 255, 120), 2)
        x = 42
        for k, v in sorted(ctrl.classifier.counts().items()):
            chip = f"{k}:{v}"
            size = cv2.getTextSize(chip, FONT, 0.55, 2)[0]
            cv2.rectangle(frame, (x - 8, y + 14), (x + size[0] + 8, y + 42),
                          (40, 90, 40), -1)
            cv2.putText(frame, chip, (x, y + 34), FONT, 0.55, (200, 255, 200), 2)
            x += size[0] + 26
        if x == 42:
            cv2.putText(frame, "(no samples yet)", (42, y + 34), FONT, 0.55,
                        (160, 200, 160), 1)
