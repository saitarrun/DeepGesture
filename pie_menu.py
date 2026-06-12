"""
PieMenu — radial quick-action menu state machine (pure geometry, no I/O).

Lifecycle (driven by GestureController, rendered by NeonHUD):

  open    hold a fist; the controller calls open_at(palm_center)
  hover   point the index finger out of the inner dead zone; the slice under
          the fingertip's angle highlights
  select  pinch -> the hovered slice flashes, the controller fires its hotkey
          through the ActionBus, and the menu closes after the flash
  cancel  fist again (after leaving fist), hand lost, or TIMEOUT idle seconds

Pinch-as-select is made mechanically robust two ways: hover is STICKY (the
fingertip falling back into the dead zone keeps the last hovered slice — it
is a selection cursor, not a literal position), and the controller FREEZES
hover once the pinch starts closing, because the act of pinching physically
drags the index tip toward the thumb. What you select is what you were
pointing at before your fingers moved.

Slice 0 is centered at 12 o'clock; slices proceed clockwise. Screen
coordinates (y down) are used throughout.
"""

import math

INNER_R = 60          # px dead zone around the anchor (no hover)
OUTER_R = 150         # px outer radius of the donut
TIMEOUT = 5.0         # s of IDLE (no hover change) -> auto close
FLASH_SEC = 0.35      # s the chosen slice flashes before the menu closes
OPEN_ANIM = 0.15      # s bloom-open animation (used by the HUD)


class PieMenu:
    def __init__(self, slices):
        self.slices = list(slices)         # [{"label": str, "keys": [...]}, ...]
        self.open = False
        self.anchor = (0.0, 0.0)
        self.hover = -1
        self.opened_at = 0.0
        self.touched_at = 0.0
        self.flash = None                  # (slice_idx, t_selected) or None

    # ------------------------------------------------------------------ #
    def open_at(self, anchor, now):
        self.open = True
        self.anchor = (float(anchor[0]), float(anchor[1]))
        self.hover = -1
        self.opened_at = now
        self.touched_at = now              # last hover activity (idle timeout)
        self.flash = None

    def close(self):
        self.open = False
        self.hover = -1
        self.flash = None

    # ------------------------------------------------------------------ #
    def slice_at(self, pt):
        """Slice index under a point, or -1 (dead zone / no slices)."""
        if not self.slices:
            return -1
        dx, dy = pt[0] - self.anchor[0], pt[1] - self.anchor[1]
        if math.hypot(dx, dy) < INNER_R:
            return -1
        ang = math.degrees(math.atan2(dy, dx)) % 360.0     # 0 = right, cw (y down)
        span = 360.0 / len(self.slices)
        rel = (ang + 90.0 + span / 2.0) % 360.0            # slice 0 at top
        return int(rel // span) % len(self.slices)

    def update(self, index_tip, now, freeze=False):
        """Refresh hover from the pointing fingertip; handle idle timeout.
        `freeze=True` keeps the current hover (used while a pinch is closing,
        since pinching drags the index tip). Hover is sticky: re-entering the
        dead zone keeps the last hovered slice."""
        if not self.open:
            return -1
        if self.flash is None and now - self.touched_at > TIMEOUT:
            self.close()
            return -1
        if not freeze:
            idx = self.slice_at(index_tip)
            if idx >= 0 and idx != self.hover:
                self.hover = idx
                self.touched_at = now
        return self.hover

    def try_select(self, pinching, now):
        """Pinch over a hovered slice selects it (once). Returns the slice."""
        if not self.open or self.flash is not None:
            return None
        if pinching and self.hover >= 0:
            self.flash = (self.hover, now)
            return self.slices[self.hover]
        return None

    def flash_done(self, now):
        return self.flash is not None and now - self.flash[1] > FLASH_SEC
