"""Industrial camera stream simulator.

Stands in for a physical GigE / RTSP conveyor camera. Each call to
``grab_frame`` returns a freshly rendered BGR image of a product (a circular
metal bearing) travelling on a conveyor belt, together with the ground-truth
defects that were procedurally painted onto it.

The ground truth is used two ways:
  * It lets the OpenCV pipeline be exercised against *real* pixel artefacts.
  * It seeds the mock-YOLO fallback so the system produces sensible detections
    even when no trained weights are present.

If ``QI_FRAME_DIR`` points at a directory of real images, those are streamed
instead (round-robin) — the on-the-fly synthesizer is only the default.
"""

from __future__ import annotations

import os
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

FRAME_W, FRAME_H = 640, 480
NOMINAL_RADIUS = 120  # nominal bearing radius in px


@dataclass
class GroundTruthDefect:
    label: str
    x: int
    y: int
    w: int
    h: int


@dataclass
class Frame:
    unit_id: str
    image: np.ndarray
    ground_truth: List[GroundTruthDefect] = field(default_factory=list)
    nominal_radius: int = NOMINAL_RADIUS
    actual_radius: int = NOMINAL_RADIUS


class CameraSimulator:
    """Generates a stream of synthetic conveyor frames."""

    DEFECT_TYPES = ("scratch", "dent", "missing_component", "stain", "misalignment")

    def __init__(self, defect_rate: float = 0.35, frame_dir: Optional[str] = None):
        self.defect_rate = defect_rate
        self._counter = 0
        self._frame_dir = frame_dir or os.getenv("QI_FRAME_DIR")
        self._files: List[Path] = []
        if self._frame_dir and Path(self._frame_dir).is_dir():
            self._files = sorted(
                p for p in Path(self._frame_dir).iterdir()
                if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")
            )

        # Operator-supplied source, switchable at runtime via the API.
        # Priority (highest first): user image > user video/stream > frame dir > synth.
        self._lock = threading.Lock()
        self._user_image: Optional[np.ndarray] = None
        self._user_video: Optional[cv2.VideoCapture] = None
        self._source_kind = "simulator"   # simulator | image | video | stream
        self._source_label = ""

    # ------------------------------------------------------------------ #
    # Runtime source switching
    # ------------------------------------------------------------------ #
    def set_image(self, img: np.ndarray, label: str = "uploaded image") -> None:
        """Stream a single uploaded still image on a loop."""
        with self._lock:
            self._release_video()
            self._user_image = cv2.resize(img, (FRAME_W, FRAME_H))
            self._source_kind = "image"
            self._source_label = label

    def set_video(self, path_or_url: str, kind: str = "video", label: str = "") -> bool:
        """Stream an uploaded video file or a live stream URL (RTSP/HTTP/webcam).

        Returns True if the source opened successfully.
        """
        cap = cv2.VideoCapture(path_or_url)
        if not cap.isOpened():
            cap.release()
            return False
        with self._lock:
            self._release_video()
            self._user_image = None
            self._user_video = cap
            self._source_kind = kind
            self._source_label = label or str(path_or_url)
        return True

    def reset_source(self) -> None:
        """Revert to the built-in synthetic simulator (or frame dir)."""
        with self._lock:
            self._release_video()
            self._user_image = None
            self._source_kind = "simulator"
            self._source_label = ""

    def _release_video(self) -> None:
        if self._user_video is not None:
            try:
                self._user_video.release()
            except Exception:
                pass
            self._user_video = None

    def source_info(self) -> dict:
        with self._lock:
            return {"kind": self._source_kind, "label": self._source_label}

    # ------------------------------------------------------------------ #
    def next_unit_id(self) -> str:
        self._counter += 1
        return f"U{int(time.time()) % 100000:05d}-{self._counter:04d}"

    # ------------------------------------------------------------------ #
    def grab_frame(self, defect_rate: Optional[float] = None) -> Frame:
        with self._lock:
            if self._user_image is not None:
                return Frame(unit_id=self.next_unit_id(), image=self._user_image.copy())
            if self._user_video is not None:
                img = self._read_video_frame()
                if img is not None:
                    return Frame(unit_id=self.next_unit_id(), image=img)
        if self._files:
            return self._grab_from_dir()
        return self._synthesize(defect_rate if defect_rate is not None else self.defect_rate)

    # ------------------------------------------------------------------ #
    def _read_video_frame(self) -> Optional[np.ndarray]:
        """Read the next frame, looping a finite video back to the start."""
        cap = self._user_video
        if cap is None:
            return None
        ok, img = cap.read()
        if not ok:
            # End of a finite file — rewind and retry once. Live streams that
            # genuinely drop will fall through to the simulator for this tick.
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, img = cap.read()
            if not ok:
                return None
        return cv2.resize(img, (FRAME_W, FRAME_H))

    # ------------------------------------------------------------------ #
    def _grab_from_dir(self) -> Frame:
        path = self._files[self._counter % len(self._files)]
        img = cv2.imread(str(path))
        if img is None:
            img = self._blank_belt()
        else:
            img = cv2.resize(img, (FRAME_W, FRAME_H))
        return Frame(unit_id=self.next_unit_id(), image=img)

    # ------------------------------------------------------------------ #
    def _blank_belt(self) -> np.ndarray:
        """Dark textured conveyor-belt background."""
        belt = np.full((FRAME_H, FRAME_W, 3), (38, 40, 44), dtype=np.uint8)
        # Subtle belt seams running across the conveyor.
        for x in range(0, FRAME_W, 64):
            cv2.line(belt, (x, 0), (x, FRAME_H), (30, 32, 36), 1)
        noise = np.random.randint(0, 12, (FRAME_H, FRAME_W, 3), dtype=np.uint8)
        return cv2.add(belt, noise)

    # ------------------------------------------------------------------ #
    def _draw_bearing(self, img: np.ndarray, cx: int, cy: int, radius: int) -> None:
        """Render a brushed-metal circular bearing with a bore hole."""
        # Outer body
        cv2.circle(img, (cx, cy), radius, (170, 172, 176), -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (cx, cy), radius, (120, 122, 126), 3, lineType=cv2.LINE_AA)
        # Brushed radial shading
        for r in range(radius, 0, -8):
            shade = 150 + int(35 * np.cos(r / 14.0))
            cv2.circle(img, (cx, cy), r, (shade, shade + 2, shade + 6), 1, lineType=cv2.LINE_AA)
        # Inner race + bore
        cv2.circle(img, (cx, cy), int(radius * 0.45), (95, 97, 101), -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (cx, cy), int(radius * 0.22), (55, 57, 61), -1, lineType=cv2.LINE_AA)
        # Mounting holes ("components") around the race
        for ang in range(0, 360, 60):
            hx = int(cx + radius * 0.72 * np.cos(np.radians(ang)))
            hy = int(cy + radius * 0.72 * np.sin(np.radians(ang)))
            cv2.circle(img, (hx, hy), 6, (70, 72, 76), -1, lineType=cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    def _synthesize(self, defect_rate: float) -> Frame:
        img = self._blank_belt()
        gts: List[GroundTruthDefect] = []

        # Slight positional jitter to mimic conveyor travel.
        cx = FRAME_W // 2 + random.randint(-25, 25)
        cy = FRAME_H // 2 + random.randint(-15, 15)

        inject = random.random() < defect_rate
        defect_type = random.choice(self.DEFECT_TYPES) if inject else None

        actual_radius = NOMINAL_RADIUS
        if defect_type == "dent":
            # Out-of-tolerance shrink/expand of the body.
            actual_radius = int(NOMINAL_RADIUS * random.choice([0.84, 1.16]))
        elif defect_type == "misalignment":
            cx += random.choice([-130, 130])  # part shifted off the belt centre line

        self._draw_bearing(img, cx, cy, actual_radius)

        if defect_type == "scratch":
            gts.append(self._paint_scratch(img, cx, cy, actual_radius))
        elif defect_type == "stain":
            gts.append(self._paint_stain(img, cx, cy, actual_radius))
        elif defect_type == "missing_component":
            gts.append(self._paint_missing_component(img, cx, cy, actual_radius))
        elif defect_type == "dent":
            gts.append(GroundTruthDefect("dent", cx - actual_radius, cy - actual_radius,
                                         2 * actual_radius, 2 * actual_radius))
        elif defect_type == "misalignment":
            gts.append(GroundTruthDefect("misalignment", cx - actual_radius, cy - actual_radius,
                                         2 * actual_radius, 2 * actual_radius))

        return Frame(
            unit_id=self.next_unit_id(),
            image=img,
            ground_truth=gts,
            nominal_radius=NOMINAL_RADIUS,
            actual_radius=actual_radius,
        )

    # ------------------------------------------------------------------ #
    def _paint_scratch(self, img, cx, cy, radius) -> GroundTruthDefect:
        a = random.uniform(0, np.pi)
        length = random.randint(40, int(radius * 1.4))
        x1 = int(cx + length / 2 * np.cos(a))
        y1 = int(cy + length / 2 * np.sin(a))
        x2 = int(cx - length / 2 * np.cos(a))
        y2 = int(cy - length / 2 * np.sin(a))
        cv2.line(img, (x1, y1), (x2, y2), (235, 238, 240), 2, lineType=cv2.LINE_AA)
        cv2.line(img, (x1, y1), (x2, y2), (210, 212, 215), 1, lineType=cv2.LINE_AA)
        xs, ys = sorted((x1, x2)), sorted((y1, y2))
        return GroundTruthDefect("scratch", xs[0] - 4, ys[0] - 4,
                                 (xs[1] - xs[0]) + 8, (ys[1] - ys[0]) + 8)

    def _paint_stain(self, img, cx, cy, radius) -> GroundTruthDefect:
        sx = cx + random.randint(-radius // 2, radius // 2)
        sy = cy + random.randint(-radius // 2, radius // 2)
        sr = random.randint(14, 26)
        overlay = img.copy()
        # Saturated rust/oil colour so HSV saturation thresholding catches it.
        colour = random.choice([(20, 70, 160), (30, 120, 60), (140, 40, 30)])
        cv2.circle(overlay, (sx, sy), sr, colour, -1, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
        return GroundTruthDefect("stain", sx - sr, sy - sr, 2 * sr, 2 * sr)

    def _paint_missing_component(self, img, cx, cy, radius) -> GroundTruthDefect:
        ang = random.choice(range(0, 360, 60))
        hx = int(cx + radius * 0.72 * np.cos(np.radians(ang)))
        hy = int(cy + radius * 0.72 * np.sin(np.radians(ang)))
        # Paint the mounting hole back over with body colour => component absent.
        cv2.circle(img, (hx, hy), 10, (170, 172, 176), -1, lineType=cv2.LINE_AA)
        return GroundTruthDefect("missing_component", hx - 12, hy - 12, 24, 24)
