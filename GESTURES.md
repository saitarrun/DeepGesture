# Gesture Reference

Complete vocabulary of built-in gestures and their default macOS actions.

## Hand pose detection

Gestures are classified per-frame by `classify_pose()` in `virtual_mouse.py`. A pose must be held for `STABLE_FRAMES` (default: 3) consecutive frames before the associated action fires, preventing accidental triggering during transitions.

---

## Built-in gestures

| Pose | Hand shape | Action | Configurable |
|------|-----------|--------|--------------|
| **index** | Index finger extended, others closed | Left click (on pose entry) | No |
| **two** | Index + middle extended | Cursor movement (2-finger midpoint) | No |
| **three** | Index + middle + ring extended | Volume control (raise/lower hand) | No |
| **palm** | All four fingers extended | Swipe left/right → Space switch | Via `swipe` config |
| **fist** | All fingers closed | Mouse hold / drag | No |
| **rock** | Index + pinky extended | Configurable macro | Via `macros.rock` |
| **pinky** | Pinky only extended | Configurable macro | Via `macros.pinky` |
| **two (tips touching)** | Index + middle extended, tips close | Right click | No |

---

## Swipe gesture

Requires open palm (all fingers up) moving horizontally across at least `SWIPE_DIST_FRAC` (30%) of frame width within `SWIPE_WINDOW` (0.5s). Vertical drift must be below `SWIPE_VERTICAL_FRAC` (15%) of frame width.

Default bindings (overridable per app in `config.json`):

```json
"swipe": {
  "next": ["ctrl", "right"],
  "prev":  ["ctrl", "left"]
}
```

Per-app override example (Google Chrome):
```json
"app_profiles": {
  "Google Chrome": {
    "swipe": {
      "next": ["command", "shift", "]"],
      "prev":  ["command", "shift", "["]
    }
  }
}
```

---

## Radial pie menu

Hold fist for `MENU_HOLD_SEC` (0.8s) to open. Point with index finger to hover a slice. Pinch to select.

Default menu items (configurable via `menu` array in `config.json`):

| Slice | Label | Keys |
|-------|-------|------|
| 0 (up) | Copy | ⌘C |
| 1 | Paste | ⌘V |
| 2 | Screenshot | ⌘⇧3 |
| 3 (down) | Mission Control | ⌃↑ |
| 4 | Spotlight | ⌘Space |
| 5 | Undo | ⌘Z |

---

## Custom ML gestures

Train your own gestures via the UI (training mode toggle). Up to `N` custom gestures can be registered, each bound to a key sequence in `gesture_actions`.

Recognition uses a deep MLP classifier (`GestureClassifier` in `gesture_ml.py`) with softmax confidence gating:
- Returns label only when confidence ≥ `gesture_threshold` (default: 0.85)
- Rejects ambiguous predictions when top-1 vs top-2 margin < 0.20

```json
"gesture_threshold": 0.85,
"gesture_actions": {
  "g1": ["command", "c"],
  "g2": ["command", "v"]
}
```

---

## Volume control

Three-finger pose (index + middle + ring). Raise hand to increase, lower to decrease.

Accumulator-based: slow precise movements fire the same number of steps as fast flicks. Deadzone of `VOLUME_DEADZONE_PX` rejects standstill jitter. Scale-invariant (works at any distance from camera).

---

## Adding new built-in gestures

1. Add a new case to `classify_pose()` in `virtual_mouse.py`
2. Add dispatch branch in `GestureController.process()`
3. Write a test in `test_virtual_mouse.py` using synthetic landmarks (see existing tests)
4. Document here

For dynamic (trajectory-based) gestures, see issue #3 (transformer encoder).
