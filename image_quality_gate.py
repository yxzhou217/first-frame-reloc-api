"""Experimental CPU precheck for relocalization input images.

Requires numpy and OpenCV >= 4.11. Scores measure visible feature support,
not probability of correct localization. Thresholds require camera-specific
calibration. Pass BGR uint8 arrays, already corrected for EXIF orientation.
"""
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class QualityConfig:
    width: int = 480
    cols: int = 6
    rows: int = 4
    corner_cap: int = 4
    min_corner_strength: float = 0.0001
    min_line_length: float = 35.0
    min_line_pixels_per_cell: int = 12
    # Experimental threshold, selected while inspecting the current probes.
    retry_threshold: float = 40.0


class ImageQualityGate:
    def __init__(self, config=None):
        self.config = config or QualityConfig()
        self.lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_NONE)

    def evaluate(self, bgr):
        c = self.config
        if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
            raise ValueError("Expected a nonempty uint8 HxWx3 BGR image")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        height = max(c.rows * 2, round(gray.shape[0] * c.width / gray.shape[1]))
        gray = cv2.resize(gray, (c.width, height), interpolation=cv2.INTER_AREA)
        gray = cv2.GaussianBlur(gray, (3, 3), 0.7)
        h, w = gray.shape

        points, strengths = cv2.goodFeaturesToTrackWithQuality(
            gray, maxCorners=800, qualityLevel=0.01, minDistance=6,
            mask=None, blockSize=5,
        )
        counts = np.zeros(c.rows * c.cols, dtype=np.int32)
        if points is not None:
            xy = points[:, 0]
            valid = ((xy[:, 0] >= 5) & (xy[:, 0] < w - 5)
                     & (xy[:, 1] >= 5) & (xy[:, 1] < h - 5)
                     & (strengths.ravel() >= c.min_corner_strength))
            xy = xy[valid]
            ids = (np.clip((xy[:, 1] * c.rows / h).astype(int), 0, c.rows - 1) * c.cols
                   + np.clip((xy[:, 0] * c.cols / w).astype(int), 0, c.cols - 1))
            counts = np.bincount(ids, minlength=c.rows * c.cols)
        corner_support = float(np.minimum(counts / c.corner_cap, 1).mean())

        # Complement sparse corners with long, spatially distributed lines.
        # A field of parallel slats should not get the same support as a
        # layout with multiple line directions.
        lines = self.lsd.detect(gray)[0]
        mask = np.zeros_like(gray)
        angles, weights = [], []
        if lines is not None:
            for x0, y0, x1, y1 in lines[:, 0]:
                length = float(np.hypot(x1 - x0, y1 - y0))
                if length < c.min_line_length:
                    continue
                if (max(x0, x1) < 5 or min(x0, x1) > w - 5
                        or max(y0, y1) < 5 or min(y0, y1) > h - 5):
                    continue
                angles.append(float(np.arctan2(y1 - y0, x1 - x0) % np.pi))
                weights.append(length)
                cv2.line(mask, (round(x0), round(y0)), (round(x1), round(y1)), 255, 1)
        secondary_share = 0.0
        if weights:
            a, weights = np.array(angles), np.array(weights)
            gap = np.abs(a[:, None] - a[None, :])
            gap = np.minimum(gap, np.pi - gap)
            parallel = gap < np.deg2rad(20)
            dominant = int(np.argmax(parallel @ weights))
            secondary_share = float(weights[~parallel[dominant]].sum() / weights.sum())
        line_pixels = np.array([
            np.count_nonzero(mask[y*h//c.rows:(y+1)*h//c.rows,
                                  x*w//c.cols:(x+1)*w//c.cols])
            for y in range(c.rows) for x in range(c.cols)
        ])
        line_coverage = float((line_pixels >= c.min_line_pixels_per_cell).mean())
        line_support = line_coverage * min(1.0, 4.0 * secondary_share)

        score = float(100.0 * max(corner_support, line_support))
        return {
            "score": round(score, 2),
            "action": "turn_and_retake" if score < c.retry_threshold else "try_localization",
            "corner_score": round(100 * corner_support, 2),
            "line_score": round(100 * line_support, 2),
            "corner_count": int(counts.sum()),
            "corner_grid": counts.reshape(c.rows, c.cols).tolist(),
            "line_coverage": round(line_coverage, 3),
            "nonparallel_line_share": round(secondary_share, 3),
        }


if __name__ == "__main__":
    import argparse
    import json
    import time

    parser = argparse.ArgumentParser(description="Experimental CPU relocalization image precheck")
    parser.add_argument("image", help="Input image path (OpenCV loads as BGR)")
    parser.add_argument("--threshold", type=float, default=40.0,
                        help="Experimental retake threshold; calibrate for the robot camera")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    image = cv2.imread(args.image)
    if image is None:
        parser.error("Image could not be decoded")
    gate = ImageQualityGate(QualityConfig(retry_threshold=args.threshold))
    start = time.perf_counter()
    result = gate.evaluate(image)
    result["evaluation_ms"] = round(1000 * (time.perf_counter() - start), 2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
