"""
Regression test suite — runs fully headless (no camera, no GUI, no real
OS input). PyAutoGUI is stubbed so every click/scroll/hotkey is recorded
and asserted instead of executed.

Run:  python test_virtual_mouse.py        (or: python -m unittest -v)
"""

import json
import math
import os
import tempfile
import unittest

import numpy as np

import virtual_mouse as vm
from gesture_ml import GestureClassifier, landmark_features
from threaded_capture import Metrics


# --------------------------------------------------------------------------- #
# PyAutoGUI stub — records calls instead of injecting OS input
# --------------------------------------------------------------------------- #
class PyAutoStub:
    def __init__(self):
        self.calls = []

    def _rec(self, name):
        def f(*a, **kw):
            self.calls.append((name, a))
        return f

    def __getattr__(self, name):
        return self._rec(name)

    def named(self, name):
        return [c for c in self.calls if c[0] == name]


# --------------------------------------------------------------------------- #
# Synthetic hands
# --------------------------------------------------------------------------- #
def make_hand(fingers=(True, False, False, False), pinch_gap=1.5,
              s=100.0, cx=600.0, cy=550.0):
    """Build a synthetic 21-landmark hand, wrist at (cx, cy), fingers up.

    fingers   = (index, middle, ring, pinky) extension flags
    pinch_gap = thumb-tip to index-tip distance as a RATIO of hand scale
    s         = hand scale in px (wrist -> middle MCP distance)
    """
    pts = [(cx, cy)] * 21
    cols = [(5, -0.45), (9, 0.0), (13, 0.45), (17, 0.9)]
    for (mcp, xo), ext in zip(cols, fingers):
        x = cx + xo * s
        pts[mcp] = (x, cy - 1.0 * s)
        pts[mcp + 1] = (x, cy - 1.5 * s)                       # PIP
        pts[mcp + 2] = (x, cy - (1.75 if ext else 1.3) * s)    # DIP
        pts[mcp + 3] = (x, cy - (2.0 if ext else 1.15) * s)    # TIP
    # thumb chain; tip placed pinch_gap * s from the index tip
    pts[1] = (cx - 0.4 * s, cy - 0.2 * s)
    pts[2] = (cx - 0.7 * s, cy - 0.5 * s)
    pts[3] = (cx - 0.8 * s, cy - 0.8 * s)
    ix, iy = pts[vm.INDEX_TIP]
    pts[4] = (ix + pinch_gap * s, iy)
    return pts


def rotate_hand(lm, deg):
    """Rotate every landmark around the wrist."""
    ox, oy = lm[0]
    a = math.radians(deg)
    c, s_ = math.cos(a), math.sin(a)
    return [(ox + (x - ox) * c - (y - oy) * s_,
             oy + (x - ox) * s_ + (y - oy) * c) for x, y in lm]


def jitter(lm, amp=2.0, seed=0):
    rng = np.random.default_rng(seed)
    return [(x + rng.uniform(-amp, amp), y + rng.uniform(-amp, amp))
            for x, y in lm]


def fresh_controller(stub):
    vm.pyautogui = stub
    cfg = json.loads(json.dumps(vm.DEFAULT_CONFIG))
    ctrl = vm.GestureController(1920, 1080, cfg)
    ctrl.classifier.clear()
    return ctrl


FRAME = lambda: np.zeros((720, 1280, 3), np.uint8)
REGION = (0, 0, 1280, 720)


# --------------------------------------------------------------------------- #
class TestGeometry(unittest.TestCase):
    def test_fingers_up_poses(self):
        self.assertEqual(vm.fingers_up(make_hand((1, 0, 0, 0))), (True, False, False, False))
        self.assertEqual(vm.fingers_up(make_hand((1, 1, 1, 1))), (True, True, True, True))
        self.assertEqual(vm.fingers_up(make_hand((0, 0, 0, 0))), (False, False, False, False))

    def test_fingers_up_rotation_invariant(self):
        for deg in (-40, -20, 25, 45):
            hand = rotate_hand(make_hand((1, 1, 0, 0)), deg)
            self.assertEqual(vm.fingers_up(hand), (True, True, False, False),
                             f"failed at {deg} deg tilt")

    def test_classify_pose(self):
        cases = {
            (1, 0, 0, 0): "index", (1, 1, 0, 0): "two", (1, 1, 1, 0): "three",
            (1, 1, 1, 1): "palm", (0, 0, 0, 0): "fist", (1, 0, 0, 1): "rock",
            (0, 0, 0, 1): "pinky", (0, 1, 0, 0): "other",
        }
        for flags, want in cases.items():
            self.assertEqual(vm.classify_pose(make_hand(flags)), want)

    def test_pinch_scale_invariant(self):
        for s in (60.0, 100.0, 180.0):
            self.assertTrue(vm.is_pinching(make_hand(pinch_gap=0.2, s=s)),
                            f"closed pinch missed at hand scale {s}")
            self.assertFalse(vm.is_pinching(make_hand(pinch_gap=1.0, s=s)),
                             f"open hand mis-read as pinch at scale {s}")


class TestPinchStateMachine(unittest.TestCase):
    def _move(self, ctrl, gap, t):
        return ctrl._move(make_hand((1, 0, 0, 0), pinch_gap=gap), REGION, t, FRAME())

    def test_quick_pinch_is_one_click(self):
        stub = PyAutoStub()
        ctrl = fresh_controller(stub)
        t = 1000.0
        for _ in range(3):                       # settle open
            self._move(ctrl, 1.5, t); t += 0.015
        for _ in range(4):                       # close
            self._move(ctrl, 0.1, t); t += 0.015
        for _ in range(3):                       # release
            self._move(ctrl, 1.5, t); t += 0.015
        self.assertEqual(len(stub.named("click")), 1)
        self.assertEqual(len(stub.named("mouseDown")), 0)

    def test_hysteresis_dead_band_no_phantom_release(self):
        stub = PyAutoStub()
        ctrl = fresh_controller(stub)
        t = 1000.0
        for _ in range(3):
            self._move(ctrl, 1.5, t); t += 0.015
        for _ in range(3):
            self._move(ctrl, 0.1, t); t += 0.015
        self.assertTrue(ctrl.pinching)
        # wobble INTO the dead band (0.47 is above PINCH_ON=0.40 but below
        # PINCH_OFF=0.55): a single-threshold design would flicker/release
        for _ in range(6):
            self._move(ctrl, 0.47, t); t += 0.015
        self.assertTrue(ctrl.pinching, "hysteresis failed: released in dead band")
        self.assertEqual(len(stub.named("click")), 0)
        for _ in range(3):                       # genuine open -> the one click
            self._move(ctrl, 1.5, t); t += 0.015
        self.assertEqual(len(stub.named("click")), 1)

    def test_held_pinch_becomes_drag(self):
        stub = PyAutoStub()
        ctrl = fresh_controller(stub)
        t = 1000.0
        for _ in range(3):
            self._move(ctrl, 1.5, t); t += 0.05
        for _ in range(10):                      # hold ~0.5s > DRAG_HOLD_SEC
            self._move(ctrl, 0.1, t); t += 0.05
        for _ in range(3):
            self._move(ctrl, 1.5, t); t += 0.05
        self.assertEqual(len(stub.named("mouseDown")), 1)
        self.assertEqual(len(stub.named("mouseUp")), 1)
        self.assertEqual(len(stub.named("click")), 0)


class TestScroll(unittest.TestCase):
    def test_fractional_ticks_accumulate(self):
        stub = PyAutoStub()
        ctrl = fresh_controller(stub)
        t, cy = 1000.0, 550.0
        for _ in range(5):                       # hand moves up 7 px / frame
            hand = make_hand((1, 1, 0, 0), cy=cy)
            ctrl._scroll_or_rclick(hand, t, FRAME())
            cy -= 7.0; t += 0.02
        total = sum(c[1][0] for c in stub.named("scroll"))
        # 4 deltas x 7px x 0.5 = 14.0 ticks; int-truncation per frame would lose some
        self.assertEqual(total, 14)

    def test_deadzone_rejects_jitter(self):
        stub = PyAutoStub()
        ctrl = fresh_controller(stub)
        t, cy = 1000.0, 550.0
        for i in range(10):                      # +-2 px wobble only
            hand = make_hand((1, 1, 0, 0), cy=cy + (2 if i % 2 else -2))
            ctrl._scroll_or_rclick(hand, t, FRAME())
            t += 0.02
        self.assertEqual(stub.named("scroll"), [])

    def test_touching_tips_is_right_click(self):
        stub = PyAutoStub()
        ctrl = fresh_controller(stub)
        hand = make_hand((1, 1, 0, 0))
        hand[vm.MIDDLE_TIP] = (hand[vm.INDEX_TIP][0] + 10, hand[vm.INDEX_TIP][1])
        ctrl._scroll_or_rclick(hand, 1000.0, FRAME())
        self.assertEqual(len(stub.named("rightClick")), 1)


class TestSwipe(unittest.TestCase):
    def test_palm_swipe_switches_space(self):
        stub = PyAutoStub()
        ctrl = fresh_controller(stub)
        t, cx = 1000.0, 300.0
        for _ in range(8):                       # 400 px right in 0.28 s
            ctrl._swipe(make_hand((1, 1, 1, 1), cx=cx), t, FRAME())
            cx += 50.0; t += 0.04
        hot = stub.named("hotkey")
        self.assertTrue(any(c[1] == ("ctrl", "right") for c in hot))

    def test_vertical_motion_rejected(self):
        stub = PyAutoStub()
        ctrl = fresh_controller(stub)
        t, cx, cy = 1000.0, 300.0, 200.0
        for _ in range(8):                       # diagonal: too much vertical
            ctrl._swipe(make_hand((1, 1, 1, 1), cx=cx, cy=cy), t, FRAME())
            cx += 50.0; cy += 30.0; t += 0.04
        self.assertEqual(stub.named("hotkey"), [])


class TestOneEuro(unittest.TestCase):
    def test_damps_jitter_when_still(self):
        f = vm.OneEuroFilter(60, min_cutoff=0.6, beta=0.03)
        out = [f(100.0 + (2.0 if i % 2 else -2.0), i / 60.0) for i in range(120)]
        tail = out[-30:]
        self.assertLess(max(tail) - min(tail), 1.0)       # +-2px noise squashed

    def test_follows_fast_motion(self):
        f = vm.OneEuroFilter(60, min_cutoff=0.6, beta=0.03)
        for i in range(30):
            f(0.0, i / 60.0)
        out = 0.0
        for i in range(30, 90):
            out = f(500.0, i / 60.0)
        self.assertGreater(out, 450.0)


class TestGestureML(unittest.TestCase):
    def test_feature_invariance(self):
        base = make_hand((0, 1, 0, 0))
        ref = landmark_features(base)
        moved = [(x + 313, y - 99) for x, y in base]
        scaled = [(600 + (x - 600) * 2.3, 550 + (y - 550) * 2.3) for x, y in base]
        rotated = rotate_hand(base, 33)
        for variant in (moved, scaled, rotated):
            self.assertTrue(np.allclose(ref, landmark_features(variant), atol=1e-6))

    def test_knn_separates_and_rejects(self):
        clf = GestureClassifier(threshold=0.85)
        a, b = make_hand((0, 1, 0, 0)), make_hand((1, 0, 0, 1))
        for i in range(10):
            clf.add("g1", landmark_features(jitter(a, seed=i)))
            clf.add("g2", landmark_features(jitter(b, seed=100 + i)))
        lab, d = clf.predict(landmark_features(jitter(a, seed=999)))
        self.assertEqual(lab, "g1")
        self.assertLess(d, 0.85)
        lab, _ = clf.predict(landmark_features(make_hand((1, 1, 1, 1))))
        self.assertEqual(lab, "unknown")

    def test_custom_gesture_fires_hotkey_once(self):
        stub = PyAutoStub()
        ctrl = fresh_controller(stub)
        trained = make_hand((0, 1, 0, 0))
        for i in range(8):
            ctrl.observe(jitter(trained, seed=i))
            ctrl.capture_sample(1)               # -> class g1 (bound to cmd+c)
        t = 1000.0
        for i in range(6):                       # hold the pose
            ctrl.observe(jitter(trained, seed=50 + i))
            ctrl._custom(t); t += 0.03
        hot = stub.named("hotkey")
        self.assertEqual(hot, [("hotkey", ("command", "c"))])


class TestActionBus(unittest.TestCase):
    def test_cooldown_blocks_rapid_refire(self):
        from actions import ActionBus
        stub = PyAutoStub()
        bus = ActionBus(stub)
        self.assertTrue(bus.click(now=1000.0, cooldown=0.3))
        self.assertFalse(bus.click(now=1000.1, cooldown=0.3))   # too soon
        self.assertTrue(bus.click(now=1000.5, cooldown=0.3))
        self.assertEqual(len(stub.named("click")), 2)

    def test_events_recorded_for_toasts(self):
        from actions import ActionBus
        bus = ActionBus(PyAutoStub())
        bus.hotkey(["command", "c"], key="x", now=1000.0)
        bus.note("PROFILE", "CHROME", now=1000.1)
        labels = [e["label"] for e in bus.events]
        self.assertEqual(labels, ["COMMAND+C", "PROFILE"])

    def test_scroll_aggregates_into_one_event(self):
        from actions import ActionBus
        stub = PyAutoStub()
        bus = ActionBus(stub)
        bus.scroll(3, now=1000.0)
        bus.scroll(4, now=1000.2)
        bus.scroll(-2, now=1000.4)
        self.assertEqual(len(stub.named("scroll")), 3)          # OS got every tick
        self.assertEqual(len(bus.events), 1)                    # HUD got one toast
        self.assertEqual(bus.events[0]["detail"], "+5")


class TestPieMenu(unittest.TestCase):
    def _fist(self, **kw):
        return make_hand((0, 0, 0, 0), pinch_gap=1.5, **kw)

    def test_fist_hold_opens_menu(self):
        stub = PyAutoStub()
        ctrl = fresh_controller(stub)
        t = 1000.0
        for _ in range(14):                      # > MENU_HOLD_SEC of fist
            pose, gesture = ctrl.process(self._fist(), REGION, t, FRAME())
            t += 0.1
        self.assertTrue(ctrl.menu.open)
        self.assertEqual(pose, "menu")

    def test_slice_angles(self):
        from pie_menu import PieMenu
        menu = PieMenu([{"label": f"s{i}", "keys": ["a"]} for i in range(6)])
        menu.open_at((600.0, 500.0), 1000.0)
        self.assertEqual(menu.slice_at((600, 350)), 0)    # straight up
        self.assertEqual(menu.slice_at((750, 500)), 2)    # right = 90deg cw boundary
        self.assertEqual(menu.slice_at((600, 650)), 3)    # straight down
        self.assertEqual(menu.slice_at((450, 500)), 5)    # left = 270deg cw boundary
        self.assertEqual(menu.slice_at((460, 560)), 4)    # lower-left interior
        self.assertEqual(menu.slice_at((600, 510)), -1)   # dead zone

    def test_point_and_pinch_fires_slice_once(self):
        stub = PyAutoStub()
        ctrl = fresh_controller(stub)
        ctrl.menu.open_at((600.0, 500.0), 1000.0)
        ctrl._menu_left_fist = False
        t = 1000.1
        # point straight up (slice 0 = Copy), open hand (no pinch) to hover
        point = make_hand((1, 0, 0, 0), pinch_gap=1.5, cx=600, cy=620)
        for _ in range(3):
            ctrl.process(point, REGION, t, FRAME()); t += 0.03
        self.assertTrue(ctrl.menu.open)
        self.assertEqual(ctrl.menu.hover, 0)
        # now pinch to select
        pinch = make_hand((1, 0, 0, 0), pinch_gap=0.1, cx=600, cy=620)
        for _ in range(6):
            ctrl.process(pinch, REGION, t, FRAME()); t += 0.03
        hot = stub.named("hotkey")
        self.assertEqual(hot, [("hotkey", ("command", "c"))])   # fired exactly once
        # flash plays out, then the menu closes
        for _ in range(15):
            ctrl.process(pinch, REGION, t, FRAME()); t += 0.05
        self.assertFalse(ctrl.menu.open)

    def test_pinch_selects_even_when_tip_drops_into_dead_zone(self):
        """Pinching drags the index tip toward the thumb. Hover must be
        sticky + frozen so the pinch still selects the pointed slice."""
        stub = PyAutoStub()
        ctrl = fresh_controller(stub)
        ctrl.menu.open_at((600.0, 500.0), 1000.0)
        ctrl._menu_left_fist = False
        t = 1000.1
        point = make_hand((1, 0, 0, 0), pinch_gap=1.5, cx=600, cy=620)
        for _ in range(3):                       # hover slice 0 (Copy)
            ctrl.process(point, REGION, t, FRAME()); t += 0.03
        self.assertEqual(ctrl.menu.hover, 0)
        # pinch with the hand sagging: index tip now INSIDE the dead zone
        sag = make_hand((1, 0, 0, 0), pinch_gap=0.1, cx=600, cy=700)
        for _ in range(6):
            ctrl.process(sag, REGION, t, FRAME()); t += 0.03
        self.assertEqual(stub.named("hotkey"),
                         [("hotkey", ("command", "c"))])   # still selected Copy

    def test_menu_idle_timeout_refreshes_on_hover_activity(self):
        from pie_menu import PieMenu, TIMEOUT
        menu = PieMenu([{"label": f"s{i}", "keys": ["a"]} for i in range(4)])
        menu.open_at((600, 500), 1000.0)
        t = 1000.0 + TIMEOUT - 0.5
        menu.update((600, 350), t)               # hover activity near deadline
        menu.update((600, 350), t + 2.0)         # idle < TIMEOUT since activity
        self.assertTrue(menu.open)
        menu.update((600, 350), t + TIMEOUT + 0.1)
        self.assertFalse(menu.open)              # idle past TIMEOUT -> closed

    def test_menu_suspends_normal_gestures(self):
        stub = PyAutoStub()
        ctrl = fresh_controller(stub)
        ctrl.menu.open_at((600.0, 500.0), 1000.0)
        idx = make_hand((1, 0, 0, 0), pinch_gap=1.5)
        t = 1000.1
        for _ in range(6):
            pose, _ = ctrl.process(idx, REGION, t, FRAME()); t += 0.03
        self.assertEqual(pose, "menu")
        self.assertEqual(stub.named("moveTo"), [])   # cursor never moved

    def test_menu_times_out(self):
        from pie_menu import PieMenu, TIMEOUT
        menu = PieMenu([{"label": "x", "keys": ["a"]}])
        menu.open_at((600, 500), 1000.0)
        menu.update((700, 500), 1000.0 + TIMEOUT + 0.1)
        self.assertFalse(menu.open)


class FakeVolume:
    def __init__(self, level=50):
        self.level = level
        self.calls = []

    def adjust(self, delta):
        self.calls.append(delta)
        self.level = max(0, min(100, self.level + delta))
        return self.level


class TestVolume(unittest.TestCase):
    THREE = (1, 1, 1, 0)                     # index + middle + ring up

    def test_smooth_slow_raise_accumulates_steps(self):
        """12 px/frame is well under the old 22 px per-frame trigger — the
        accumulator must still convert it into multiple volume steps."""
        ctrl = fresh_controller(PyAutoStub())
        fake = FakeVolume()
        ctrl.volume = fake
        t, cy = 1000.0, 600.0
        for _ in range(12):
            ctrl._volume(make_hand(self.THREE, cy=cy), t)
            cy -= 12.0
            t += 0.1
        self.assertGreaterEqual(len(fake.calls), 4)
        self.assertTrue(all(d > 0 for d in fake.calls))
        self.assertGreater(fake.level, 50)

    def test_lowering_decreases(self):
        ctrl = fresh_controller(PyAutoStub())
        fake = FakeVolume()
        ctrl.volume = fake
        t, cy = 1000.0, 400.0
        for _ in range(12):
            ctrl._volume(make_hand(self.THREE, cy=cy), t)
            cy += 12.0
            t += 0.1
        self.assertTrue(fake.calls and all(d < 0 for d in fake.calls))
        self.assertLess(fake.level, 50)

    def test_standstill_jitter_does_not_drift(self):
        ctrl = fresh_controller(PyAutoStub())
        fake = FakeVolume()
        ctrl.volume = fake
        t = 1000.0
        for i in range(30):
            ctrl._volume(jitter(make_hand(self.THREE), amp=1.0, seed=i), t)
            t += 0.033
        self.assertEqual(fake.calls, [])

    def test_scale_invariant(self):
        """Same physical motion ratio fires the same steps near and far."""
        totals = []
        for s in (60.0, 150.0):
            ctrl = fresh_controller(PyAutoStub())
            fake = FakeVolume()
            ctrl.volume = fake
            t, cy = 1000.0, 600.0
            for _ in range(10):
                ctrl._volume(make_hand(self.THREE, s=s, cy=cy), t)
                cy -= 0.12 * s               # 12% of hand scale per frame
                t += 0.1
            totals.append(len(fake.calls))
        self.assertEqual(totals[0], totals[1])


class FakeWatcher:
    def __init__(self, current=""):
        self.current = current


class TestAppProfiles(unittest.TestCase):
    def test_resolve_profile_merges_overrides(self):
        from app_profiles import resolve_profile
        cfg = json.loads(json.dumps(vm.DEFAULT_CONFIG))
        base = resolve_profile(cfg, "SomeOtherApp")
        self.assertEqual(base["swipe"]["next"], ["ctrl", "right"])
        chrome = resolve_profile(cfg, "Google Chrome")
        self.assertEqual(chrome["swipe"]["next"], ["command", "shift", "]"])
        self.assertEqual(chrome["macros"], base["macros"])      # untouched

    def test_controller_swaps_profile_and_toasts(self):
        stub = PyAutoStub()
        ctrl = fresh_controller(stub)
        ctrl.watcher = FakeWatcher("Google Chrome")
        ctrl.refresh_profile(1000.0)
        self.assertEqual(ctrl.app, "Google Chrome")
        self.assertEqual(ctrl.profile["swipe"]["next"], ["command", "shift", "]"])
        self.assertTrue(any(e["label"] == "PROFILE" and e["detail"] == "GOOGLE CHROME"
                            for e in ctrl.bus.events))

    def test_swipe_uses_active_profile_keys(self):
        stub = PyAutoStub()
        ctrl = fresh_controller(stub)
        ctrl.watcher = FakeWatcher("Google Chrome")
        ctrl.refresh_profile(1000.0)
        t, cx = 1000.0, 300.0
        for _ in range(8):
            ctrl._swipe(make_hand((1, 1, 1, 1), cx=cx), t, FRAME())
            cx += 50.0; t += 0.04
        hot = stub.named("hotkey")
        self.assertTrue(any(c[1] == ("command", "shift", "]") for c in hot))


class TestNeonHUD(unittest.TestCase):
    STATS = {"fps": 50.0, "infer": 8.2, "loop": 12.4, "jitter": 1.5}

    def _hud(self):
        from neon_hud import NeonHUD
        return NeonHUD(vm.PINCH_ON, vm.PINCH_OFF, vm.SWIPE_DIST_FRAC)

    def test_renders_all_poses(self):
        stub = PyAutoStub()
        ctrl = fresh_controller(stub)
        ctrl._pinch_lp.s = 0.7                   # pinch gauge has data
        hud = self._hud()
        hand = make_hand((1, 0, 0, 0))
        t = 1000.0
        for pose in ("index", "two", "three", "palm", "rock", "pinky",
                     "fist", "other", "2 hands", "no hand"):
            frame = FRAME()
            hands = [] if pose == "no hand" else [hand]
            hud.draw(frame, self.STATS, pose, "action", REGION, ctrl, hands, t)
            self.assertTrue(frame.any(), f"HUD drew nothing for pose {pose!r}")
            t += 0.033

    def test_renders_boot_and_glitch_frames(self):
        """Cyberpunk layer: boot sequence and action-glitch paths render."""
        stub = PyAutoStub()
        ctrl = fresh_controller(stub)
        ctrl._pinch_lp.s = 0.7
        hud = self._hud()
        hand = make_hand((1, 0, 0, 0))
        frame = FRAME()
        hud.draw(frame, self.STATS, "index", "move", REGION, ctrl, [hand], 1000.0)
        boot = FRAME()                            # mid-boot (t = 0.8s)
        hud.draw(boot, self.STATS, "index", "move", REGION, ctrl, [hand], 1000.8)
        self.assertTrue(boot.any())
        ctrl.bus.click(now=1003.0)                # action fires -> glitch burst
        glitch = FRAME()
        hud.draw(glitch, self.STATS, "index", "click", REGION, ctrl, [hand], 1003.02)
        self.assertLess(hud._t_prev, hud._glitch_until)   # burst is active
        self.assertTrue(glitch.any())

    def test_accent_crossfades_between_poses(self):
        from neon_hud import POSE_COLORS
        stub = PyAutoStub()
        ctrl = fresh_controller(stub)
        hud = self._hud()
        hand = make_hand((1, 0, 0, 0))
        t = 1000.0
        for _ in range(10):                       # settle on index cyan
            hud.draw(FRAME(), self.STATS, "index", "move", REGION, ctrl, [hand], t)
            t += 0.033
        hud.draw(FRAME(), self.STATS, "palm", "swipe", REGION, ctrl, [hand], t)
        mid = tuple(hud._accent)                  # one frame after the flip
        for ch, lo, hi in zip(mid, POSE_COLORS["index"], POSE_COLORS["palm"]):
            self.assertTrue(min(lo, hi) - 1 <= ch <= max(lo, hi) + 1,
                            f"accent {mid} not between index and palm")
        self.assertNotEqual(tuple(int(c) for c in mid), POSE_COLORS["palm"])

    def test_renders_training_and_calibration_overlays(self):
        stub = PyAutoStub()
        ctrl = fresh_controller(stub)
        hud = self._hud()
        hand = make_hand((1, 1, 1, 1))
        ctrl.training = True
        ctrl.observe(hand)
        ctrl.capture_sample(1)
        frame = FRAME()
        hud.draw(frame, self.STATS, "train", "sampling", REGION, ctrl, [hand], 1000.0)
        self.assertTrue(frame.any())
        ctrl.training = False
        ctrl.calibrating = True
        ctrl._cal_box = [100, 100, 400, 350]
        frame = FRAME()
        hud.draw(frame, self.STATS, "calib", "tracing", REGION, ctrl, [hand], 1000.1)
        self.assertTrue(frame.any())

    def test_trail_clears_when_pose_changes(self):
        stub = PyAutoStub()
        ctrl = fresh_controller(stub)
        ctrl._pinch_lp.s = 0.7
        hud = self._hud()
        hand = make_hand((1, 0, 0, 0))
        for i in range(5):
            hud.draw(FRAME(), self.STATS, "index", "move", REGION, ctrl,
                     [hand], 1000.0 + i * 0.033)
        self.assertGreater(len(hud.trail), 0)
        hud.draw(FRAME(), self.STATS, "fist", "neutral", REGION, ctrl,
                 [hand], 1000.2)
        self.assertEqual(len(hud.trail), 0)


class TestConfigAndMetrics(unittest.TestCase):
    def test_config_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "config.json")
            cfg = vm.load_config(p)              # creates defaults
            self.assertTrue(os.path.exists(p))
            cfg["min_cutoff"] = 0.42
            vm.save_config(cfg, p)
            self.assertEqual(vm.load_config(p)["min_cutoff"], 0.42)

    def test_metrics_fps(self):
        m = Metrics()
        for i in range(60):
            m.tick(i * 0.02)                     # 50 Hz
        self.assertAlmostEqual(m.stats()["fps"], 50.0, delta=0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
