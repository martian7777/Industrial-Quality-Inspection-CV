"""Hybrid inspection engine: classical OpenCV + YOLO11.

Pipeline per frame:
  1. Classical OpenCV detectors run high-speed checks:
       - scratches  -> directional edges (Canny + Hough)
       - stains     -> HSV saturation blobs
       - dent       -> radius deviation vs. nominal (dimensional gate)
       - missing_component / misalignment -> contour + position analysis
  2. YOLO11 (Ultralytics) runs surface-defect detection *if* weights are
     available. When they are not, a deterministic mock model derives
     plausible detections from the camera ground truth so the whole pipeline
     can be demonstrated end-to-end without a GPU or trained model.
  3. The decision merger deduplicates detections and applies the operator's
     policy (confidence, min area, max-defects-to-pass) to produce a verdict.

All detectors honour the live ``InspectionConfig`` so the Streamlit control
panel changes behaviour immediately.
"""

from __future__ import annotations

import time
from typing import List, Optional

import cv2
import numpy as np

from .camera import Frame
from .config import InspectionConfig, settings

# Defect class -> annotation colour (BGR)
DEFECT_COLOURS = {
    "scratch": (0, 215, 255),
    "dent": (0, 140, 255),
    "missing_component": (0, 0, 255),
    "stain": (200, 60, 200),
    "misalignment": (255, 200, 0),
    "unknown": (180, 180, 180),
}


class Detection(dict):
    """A detected defect (dict subclass for trivial JSON serialization)."""

    def __init__(self, x, y, w, h, label, confidence, source):
        super().__init__(
            x=int(x), y=int(y), w=int(w), h=int(h),
            label=label, confidence=round(float(confidence), 3), source=source,
        )


# --------------------------------------------------------------------------- #
# YOLO loader (lazy, optional)
# --------------------------------------------------------------------------- #
class _YoloRunner:
    """Wraps Ultralytics YOLO; degrades gracefully to a mock when unavailable."""

    def __init__(self) -> None:
        self.model = None
        self.available = False
        if settings.yolo_enabled:
            self._try_load()

    def _try_load(self) -> None:
        try:
            from ultralytics import YOLO  # noqa: WPS433 (optional heavy import)

            self.model = YOLO(settings.yolo_weights)
            self.available = True
        except Exception as exc:  # pragma: no cover - depends on local env
            print(f"[inference] YOLO unavailable, using mock fallback: {exc}")
            self.available = False

    def predict(self, frame: Frame, conf: float) -> List[Detection]:
        if self.available and self.model is not None:
            return self._predict_real(frame.image, conf)
        return self._predict_mock(frame, conf)

    # -- real ---------------------------------------------------------------- #
    def _predict_real(self, image: np.ndarray, conf: float) -> List[Detection]:
        dets: List[Detection] = []
        try:
            results = self.model.predict(image, conf=conf, verbose=False)
            for r in results:
                names = r.names
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls = names.get(int(box.cls[0]), "unknown")
                    label = cls if cls in DEFECT_COLOURS else "scratch"
                    dets.append(
                        Detection(x1, y1, x2 - x1, y2 - y1, label, float(box.conf[0]), "yolo")
                    )
        except Exception as exc:  # pragma: no cover
            print(f"[inference] YOLO predict failed: {exc}")
        return dets

    # -- mock ---------------------------------------------------------------- #
    def _predict_mock(self, frame: Frame, conf: float) -> List[Detection]:
        """Derive surface-defect detections from camera ground truth.

        Only surface defects (scratch/stain/missing_component) are emitted here;
        dent/misalignment are handled by the dimensional OpenCV gate so the two
        sources stay complementary, just like a real deployment.
        """
        surface = {"scratch", "stain", "missing_component"}
        dets: List[Detection] = []
        rng = np.random.default_rng(abs(hash(frame.unit_id)) % (2**32))
        for gt in frame.ground_truth:
            if gt.label not in surface:
                continue
            simulated_conf = float(rng.uniform(0.62, 0.94))
            if simulated_conf < conf:
                continue
            jitter = rng.integers(-3, 4, size=4)
            dets.append(
                Detection(
                    gt.x + jitter[0], gt.y + jitter[1],
                    gt.w + jitter[2], gt.h + jitter[3],
                    gt.label, simulated_conf, "yolo",
                )
            )
        return dets


# --------------------------------------------------------------------------- #
# Classical OpenCV detectors
# --------------------------------------------------------------------------- #
def _segment_part(image: np.ndarray):
    """Return (mask, largest_contour) for the metal part vs. dark belt."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask, None
    return mask, max(contours, key=cv2.contourArea)


def detect_stains(image: np.ndarray, cfg: InspectionConfig) -> List[Detection]:
    """High-saturation blobs on an otherwise desaturated metal surface."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    _, mask = cv2.threshold(sat, cfg.stain_saturation_thresh, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: List[Detection] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < cfg.min_defect_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        conf = min(0.99, 0.55 + area / 4000.0)
        out.append(Detection(x, y, w, h, "stain", conf, "opencv"))
    return out


def detect_scratches(image: np.ndarray, cfg: InspectionConfig, part_contour) -> List[Detection]:
    """Thin bright linear features via Canny + probabilistic Hough lines."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Mask to the part so belt seams don't register as scratches.
    region = np.zeros_like(gray)
    if part_contour is not None:
        cv2.drawContours(region, [part_contour], -1, 255, -1)
        region = cv2.erode(region, np.ones((9, 9), np.uint8))
        gray = cv2.bitwise_and(gray, region)

    edges = cv2.Canny(gray, cfg.scratch_canny_low, cfg.scratch_canny_high)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30,
                            minLineLength=35, maxLineGap=6)
    out: List[Detection] = []
    if lines is None:
        return out
    for ln in lines[:6]:  # cap to avoid runaway annotations
        x1, y1, x2, y2 = ln[0]
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < 35:
            continue
        x, y = min(x1, x2), min(y1, y2)
        w, h = abs(x2 - x1) or 4, abs(y2 - y1) or 4
        if w * h < cfg.min_defect_area // 2:
            continue
        conf = min(0.97, 0.5 + length / 300.0)
        out.append(Detection(x - 3, y - 3, w + 6, h + 6, "scratch", conf, "opencv"))
    return _nms(out, 0.4)


def detect_dimensional(frame: Frame, cfg: InspectionConfig, part_contour) -> List[Detection]:
    """Dimensional gate: radius deviation => dent; centroid offset => misalignment."""
    out: List[Detection] = []
    if part_contour is None:
        return out

    (cx, cy), radius = cv2.minEnclosingCircle(part_contour)
    nominal = float(frame.nominal_radius)
    dev = abs(radius - nominal) / nominal
    x, y, w, h = cv2.boundingRect(part_contour)

    if dev > cfg.dimensional_tolerance:
        conf = min(0.99, 0.6 + dev)
        out.append(Detection(x, y, w, h, "dent", conf, "opencv"))

    # Misalignment: centroid far from the frame centre line.
    frame_cx = frame.image.shape[1] / 2.0
    offset = abs(cx - frame_cx) / frame.image.shape[1]
    if offset > cfg.dimensional_tolerance * 2.5:
        conf = min(0.99, 0.55 + offset)
        out.append(Detection(x, y, w, h, "misalignment", conf, "opencv"))
    return out


# --------------------------------------------------------------------------- #
# Non-max suppression + merge
# --------------------------------------------------------------------------- #
def _iou(a: Detection, b: Detection) -> float:
    ax2, ay2 = a["x"] + a["w"], a["y"] + a["h"]
    bx2, by2 = b["x"] + b["w"], b["y"] + b["h"]
    ix1, iy1 = max(a["x"], b["x"]), max(a["y"], b["y"])
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union else 0.0


def _nms(dets: List[Detection], thresh: float) -> List[Detection]:
    dets = sorted(dets, key=lambda d: d["confidence"], reverse=True)
    keep: List[Detection] = []
    for d in dets:
        if all(_iou(d, k) < thresh or d["label"] != k["label"] for k in keep):
            keep.append(d)
    return keep


# --------------------------------------------------------------------------- #
# Annotation
# --------------------------------------------------------------------------- #
def annotate(image: np.ndarray, dets: List[Detection], passed: bool) -> np.ndarray:
    out = image.copy()
    for d in dets:
        colour = DEFECT_COLOURS.get(d["label"], DEFECT_COLOURS["unknown"])
        x, y, w, h = d["x"], d["y"], d["w"], d["h"]
        cv2.rectangle(out, (x, y), (x + w, y + h), colour, 2)
        tag = f"{d['label']} {d['confidence']:.2f} [{d['source'][:3]}]"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(out, (x, y - th - 6), (x + tw + 4, y), colour, -1)
        cv2.putText(out, tag, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (20, 20, 20), 1, cv2.LINE_AA)

    # Verdict banner
    banner = (40, 160, 40) if passed else (40, 40, 200)
    label = "PASS" if passed else "REJECT"
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), banner, -1)
    cv2.putText(out, f"  {label}", (6, 20), cv2.FONT_HERSHEY_DUPLEX, 0.7,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class InspectionResult:
    def __init__(self, frame: Frame, dets: List[Detection], passed: bool,
                 confidence: float, cycle_time_ms: float, annotated: np.ndarray):
        self.unit_id = frame.unit_id
        self.detections = dets
        self.passed = passed
        self.confidence = confidence
        self.cycle_time_ms = cycle_time_ms
        self.annotated = annotated


class InspectionEngine:
    def __init__(self) -> None:
        self.yolo = _YoloRunner()

    @property
    def yolo_active(self) -> bool:
        return self.yolo.available

    def inspect(self, frame: Frame, cfg: InspectionConfig) -> InspectionResult:
        t0 = time.perf_counter()
        dets: List[Detection] = []

        use_opencv = cfg.active_model in ("hybrid", "opencv_only")
        use_yolo = cfg.active_model in ("hybrid", "yolo_only")

        part_contour = None
        if use_opencv:
            _, part_contour = _segment_part(frame.image)
            dets += detect_stains(frame.image, cfg)
            dets += detect_scratches(frame.image, cfg, part_contour)
            dets += detect_dimensional(frame, cfg, part_contour)

        if use_yolo:
            dets += self.yolo.predict(frame, cfg.yolo_confidence)

        # Merge duplicate detections of the same class from both sources.
        dets = _nms(dets, 0.45)

        passed = len(dets) <= cfg.max_defects_to_pass
        confidence = (
            float(np.mean([d["confidence"] for d in dets])) if dets else 0.99
        )
        annotated = annotate(frame.image, dets, passed)
        cycle_ms = (time.perf_counter() - t0) * 1000.0
        return InspectionResult(frame, dets, passed, confidence, cycle_ms, annotated)
