"""
Per-app gesture profiles.

AppWatcher polls the frontmost macOS application on a daemon thread (the
osascript call takes tens of milliseconds, far too slow for the frame loop —
same latest-value pattern as ThreadedCamera). The controller compares
`watcher.current` against its active profile each frame (a cheap string
compare) and re-resolves bindings only on change.

`resolve_profile()` merges the base bindings with the app's overrides from
config.json:

    "app_profiles": {
        "Google Chrome": {
            "swipe": {"next": ["command", "shift", "]"],
                      "prev": ["command", "shift", "["]}
        }
    }

Overridable per app: "swipe" (next/prev hotkeys), "macros" (pose -> hotkey),
and "gesture_actions" (trained ML class -> hotkey).

Note: the System Events query may trigger a one-time macOS Automation
permission prompt. If denied, detection logs one warning and the app stays on
the default profile.
"""

import json
import logging
import subprocess
import threading
import time

log = logging.getLogger("deepgesture.profiles")

_OSA = ('tell application "System Events" to get name of '
        "first process whose frontmost is true")


class AppWatcher:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.current = ""                  # frontmost app name ("" = unknown)
        self._stopped = False
        self._warned = False
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stopped = True

    def _poll(self):
        while not self._stopped:
            try:
                name = subprocess.check_output(
                    ["osascript", "-e", _OSA],
                    stderr=subprocess.DEVNULL, timeout=2.0,
                ).decode("utf-8", "replace").strip()
                if name:
                    self.current = name
            except Exception as exc:
                if not self._warned:
                    log.warning("frontmost-app detection unavailable (%s); "
                                "staying on the default profile", exc)
                    self._warned = True
            time.sleep(self.interval)


def resolve_profile(config, app):
    """Base bindings <- per-app overrides. Returns a fresh dict each call."""
    base = {
        "swipe": dict(config.get("swipe") or
                      {"next": ["ctrl", "right"], "prev": ["ctrl", "left"]}),
        "macros": dict(config.get("macros") or {}),
        "gesture_actions": dict(config.get("gesture_actions") or {}),
    }
    overrides = (config.get("app_profiles") or {}).get(app)
    if overrides:
        prof = json.loads(json.dumps(base))            # deep copy
        if isinstance(overrides.get("swipe"), dict):
            prof["swipe"].update(overrides["swipe"])
        if isinstance(overrides.get("macros"), dict):
            prof["macros"].update(overrides["macros"])
        if isinstance(overrides.get("gesture_actions"), dict):
            prof["gesture_actions"].update(overrides["gesture_actions"])
        return prof
    return base
