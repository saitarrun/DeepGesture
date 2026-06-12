"""
Threaded camera capture + pipeline metrics.

`cv2.VideoCapture.read()` blocks until the next frame arrives, so on the main
thread the camera and the MediaPipe inference run strictly serially. Moving the
capture to a background thread that always holds the *latest* frame lets the
main thread grab-and-go: capture I/O overlaps with inference, and stale frames
are dropped instead of queued (lower latency, higher effective FPS).
"""

import threading
import time
from collections import deque

import cv2


class ThreadedCamera:
    def __init__(self, index, width, height, fps):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self._lock = threading.Lock()
        self._ret = False
        self._frame = None
        self._stopped = False
        self.frames_read = 0
        self._thread = threading.Thread(target=self._reader, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def isOpened(self):
        return self.cap.isOpened()

    def _reader(self):
        while not self._stopped:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.005)
                continue
            with self._lock:
                self._ret, self._frame = ret, frame
                self.frames_read += 1

    def read(self):
        """Return (ok, latest_frame_copy)."""
        with self._lock:
            if self._frame is None:
                return False, None
            return self._ret, self._frame.copy()

    def release(self):
        self._stopped = True
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self.cap.release()


class Metrics:
    """Rolling pipeline metrics: FPS, inference ms, loop ms, frame-time jitter."""

    def __init__(self, n=90):
        self.infer = deque(maxlen=n)        # ms
        self.loop = deque(maxlen=n)         # ms
        self.intervals = deque(maxlen=n)    # s
        self._last = None

    def tick(self, t):
        if self._last is not None:
            self.intervals.append(t - self._last)
        self._last = t

    def add_infer(self, ms):
        self.infer.append(ms)

    def add_loop(self, ms):
        self.loop.append(ms)

    @staticmethod
    def _mean(d):
        return sum(d) / len(d) if d else 0.0

    def stats(self):
        intervals = self.intervals
        fps = 1.0 / self._mean(intervals) if intervals and self._mean(intervals) else 0.0
        if len(intervals) > 1:
            m = self._mean(intervals)
            var = sum((x - m) ** 2 for x in intervals) / len(intervals)
            jitter = (var ** 0.5) * 1000.0
        else:
            jitter = 0.0
        return {
            "fps": fps,
            "infer": self._mean(self.infer),
            "loop": self._mean(self.loop),
            "jitter": jitter,
        }
