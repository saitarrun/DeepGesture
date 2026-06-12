"""
ActionBus — the single exit point for every OS side effect.

All clicks, scrolls, hotkeys, macros, menu picks, and volume steps flow
through `fire()`, which is the only place that:

  1. checks cooldowns (one table, consistent semantics everywhere — replaces
     the controller's scattered last_click / last_rclick / last_* timers)
  2. touches the input backend (pyautogui, injectable for headless tests)
  3. records an event for the HUD toast feed — so what the UI shows is, by
     construction, exactly what fired

Continuous actions (scroll ticks, volume steps) aggregate into a single
rolling event instead of spamming a toast per tick: while the gesture
continues, the one toast updates its running total and stays fresh.

Labels are ASCII only (OpenCV Hershey fonts cannot render non-ASCII).
"""

import time
from collections import deque


class ActionBus:
    def __init__(self, backend, max_events=8):
        self.backend = backend             # pyautogui module (or a test stub)
        self.events = deque(maxlen=max_events)
        self._last = {}                    # cooldown key -> last fire time

    # ------------------------------------------------------------------ #
    # core
    # ------------------------------------------------------------------ #
    def fire(self, key, label, fn, cooldown=0.0, detail="", kind="action",
             now=None, agg_delta=None):
        """Run `fn` if `key` is off cooldown; record a toast event.
        Returns True if it fired. With `agg_delta`, consecutive fires of the
        same label fold into one rolling event (running total in `detail`)."""
        now = time.time() if now is None else now
        if cooldown > 0.0 and now - self._last.get(key, -1e9) < cooldown:
            return False
        fn()
        self._last[key] = now
        if agg_delta is not None and self.events:
            ev = self.events[-1]
            if ev["label"] == label and now - ev["t"] < 1.0:
                ev["total"] += agg_delta
                ev["detail"] = f"{ev['total']:+d}"
                ev["t"] = now
                return True
        self._push(label, detail if agg_delta is None else f"{agg_delta:+d}",
                   kind, now, total=agg_delta or 0)
        return True

    def note(self, label, detail="", kind="info", now=None):
        """Record a toast with no OS side effect (e.g. profile switch)."""
        self._push(label, detail, kind, time.time() if now is None else now)

    def _push(self, label, detail, kind, now, total=0):
        self.events.append({"t": now, "label": label, "detail": detail,
                            "kind": kind, "total": total})

    # ------------------------------------------------------------------ #
    # one-shots
    # ------------------------------------------------------------------ #
    def click(self, now=None, cooldown=0.0):
        return self.fire("click", "CLICK",
                         lambda: self.backend.click(_pause=False),
                         cooldown, kind="mouse", now=now)

    def rclick(self, now=None, cooldown=0.0):
        return self.fire("rclick", "R-CLICK",
                         lambda: self.backend.rightClick(_pause=False),
                         cooldown, kind="mouse", now=now)

    def hotkey(self, keys, label=None, key="hotkey", cooldown=0.0,
               detail="", now=None):
        keys = list(keys)
        label = label or "+".join(keys).upper()
        return self.fire(key, label,
                         lambda: self.backend.hotkey(*keys),
                         cooldown, detail=detail, kind="key", now=now)

    # ------------------------------------------------------------------ #
    # continuous / silent
    # ------------------------------------------------------------------ #
    def move(self, x, y):
        """Cursor motion — silent (a toast per frame would be noise)."""
        self.backend.moveTo(x, y, _pause=False)

    def mouse_down(self, now=None):
        self.backend.mouseDown(_pause=False)
        self._push("DRAG START", "", "mouse", time.time() if now is None else now)

    def mouse_up(self, now=None, quiet=False):
        self.backend.mouseUp(_pause=False)
        if not quiet:
            self._push("DRAG END", "", "mouse",
                       time.time() if now is None else now)

    def scroll(self, ticks, now=None):
        return self.fire("scroll", "SCROLL",
                         lambda: self.backend.scroll(ticks, _pause=False),
                         kind="mouse", now=now, agg_delta=ticks)

    def hscroll(self, ticks, now=None):
        return self.fire("hscroll", "H-SCROLL",
                         lambda: self.backend.hscroll(ticks, _pause=False),
                         kind="mouse", now=now, agg_delta=ticks)
