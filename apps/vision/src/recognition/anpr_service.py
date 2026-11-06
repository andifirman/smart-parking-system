"""End-to-end license-plate detection and recognition service."""

from __future__ import annotations

import base64
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter

import cv2
import numpy as np
import torch  # Load Torch DLLs before Paddle on Windows.
import paddle
from paddleocr import TextRecognition
from ultralytics import YOLO

PLATE_PATTERN = re.compile(r"^[A-Z]{1,3}[0-9]{1,4}[A-Z]{0,3}$")


def normalize_plate_text(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def decision_status(
    text: str,
    confidence: float,
    *,
    accept_threshold: float,
    review_threshold: float,
) -> str:
    if PLATE_PATTERN.fullmatch(text) and confidence >= accept_threshold:
        return "ACCEPT"
    if text and confidence >= review_threshold:
        return "REVIEW"
    return "REJECT"


def candidate_score(text: str, confidence: float, votes: int = 1) -> float:
    """Rank OCR candidates using confidence, syntax, and cross-variant agreement."""
    score = float(confidence)
    score += 0.25 if PLATE_PATTERN.fullmatch(text) else 0.0
    score += 0.05 if any(char.isalpha() for char in text) and any(char.isdigit() for char in text) else 0.0
    score += 0.12 * max(0, votes - 1)
    return score - (0.20 if not 3 <= len(text) <= 10 else 0.0)


def plate_preprocessing_variants(crop: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Create complementary OCR inputs while preserving the original crop."""
    scale = max(1.0, 112 / max(crop.shape[0], 1))
    resized = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if resized.ndim == 3 else resized
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        clahe,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7,
    )
    variants = [("original", resized), ("clahe", clahe), ("otsu", otsu), ("adaptive", adaptive)]
    rectified = rectify_plate_crop(resized)
    if rectified is not None:
        rectified_gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
        rectified_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(rectified_gray)
        variants.extend([("rectified", rectified), ("rectified_clahe", rectified_clahe)])
    return variants


def rectify_plate_crop(crop: np.ndarray) -> np.ndarray | None:
    """Find a plausible plate quadrilateral and warp it to a frontal view."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 140)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    crop_area = crop.shape[0] * crop.shape[1]
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        area = cv2.contourArea(polygon)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon) or area < crop_area * 0.18:
            continue
        points = polygon.reshape(4, 2).astype(np.float32)
        ordered = _order_quad_points(points)
        top_left, top_right, bottom_right, bottom_left = ordered
        width = int(max(np.linalg.norm(top_right - top_left), np.linalg.norm(bottom_right - bottom_left)))
        height = int(max(np.linalg.norm(bottom_left - top_left), np.linalg.norm(bottom_right - top_right)))
        if height < 16 or width / max(height, 1) < 1.7:
            continue
        destination = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
        matrix = cv2.getPerspectiveTransform(ordered, destination)
        return cv2.warpPerspective(crop, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
    return None


def _order_quad_points(points: np.ndarray) -> np.ndarray:
    ordered = np.zeros((4, 2), dtype=np.float32)
    coordinate_sum = points.sum(axis=1)
    coordinate_difference = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(coordinate_sum)]
    ordered[2] = points[np.argmax(coordinate_sum)]
    ordered[1] = points[np.argmin(coordinate_difference)]
    ordered[3] = points[np.argmax(coordinate_difference)]
    return ordered


def encode_jpeg(image: np.ndarray, quality: int = 90) -> str:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("Failed to encode result image.")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


@dataclass(frozen=True)
class AnprSettings:
    detector_path: Path
    recognizer_dir: Path
    detection_threshold: float = 0.40
    accept_threshold: float = 0.90
    review_threshold: float = 0.75
    image_size: int = 960


class AnprService:
    """Load models once and process uploaded images without persisting them."""

    def __init__(self, settings: AnprSettings) -> None:
        self.settings = settings
        self._inference_lock = Lock()
        self._validate_assets()
        self._paddle_device = "gpu:0" if paddle.device.is_compiled_with_cuda() else "cpu"
        self._yolo_device: int | str = 0 if torch.cuda.is_available() else "cpu"
        self._detector = YOLO(str(settings.detector_path))
        self._recognizer = TextRecognition(
            model_name="en_PP-OCRv5_mobile_rec",
            model_dir=str(settings.recognizer_dir),
            device=self._paddle_device,
        )

    def _validate_assets(self) -> None:
        if not self.settings.detector_path.is_file():
            raise FileNotFoundError(f"YOLO detector not found: {self.settings.detector_path}")
        recognizer_config = self.settings.recognizer_dir / "inference.yml"
        if not recognizer_config.is_file():
            raise FileNotFoundError(f"PaddleOCR model not found: {recognizer_config}")

    @property
    def devices(self) -> dict[str, str]:
        return {"detector": str(self._yolo_device), "recognizer": self._paddle_device}

    def analyze_bytes(self, content: bytes) -> dict[str, object]:
        image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError("The uploaded file is not a readable image.")
        with self._inference_lock:
            return self._analyze_image(image)

    def _analyze_image(self, image: np.ndarray) -> dict[str, object]:
        started_at = perf_counter()
        prediction = self._detector.predict(
            image,
            conf=self.settings.detection_threshold,
            imgsz=self.settings.image_size,
            device=self._yolo_device,
            verbose=False,
        )[0]
        annotated = image.copy()
        detections: list[dict[str, object]] = []
        boxes = prediction.boxes if prediction.boxes is not None else []
        ordered_boxes = sorted(boxes, key=lambda box: float(box.xyxy[0][0]))

        for index, box in enumerate(ordered_boxes, start=1):
            crop, coordinates = self._crop_with_margin(image, box.xyxy[0].tolist())
            if crop.size == 0:
                continue
            text, ocr_confidence, ocr_variant, ocr_latency_ms, candidates = self._recognize_crop(crop)
            status = decision_status(
                text,
                ocr_confidence,
                accept_threshold=self.settings.accept_threshold,
                review_threshold=self.settings.review_threshold,
            )
            detection_confidence = float(box.conf[0])
            detections.append(
                {
                    "index": index,
                    "plate_text": text,
                    "status": status,
                    "detection_confidence": detection_confidence,
                    "ocr_confidence": ocr_confidence,
                    "ocr_latency_ms": round(ocr_latency_ms, 1),
                    "ocr_variant": ocr_variant,
                    "ocr_candidates": candidates,
                    "bounding_box": list(coordinates),
                    "crop_image_base64": encode_jpeg(crop),
                }
            )
            self._draw_detection(annotated, coordinates, text, status, ocr_confidence)

        return {
            "detections": detections,
            "detection_count": len(detections),
            "processing_ms": round((perf_counter() - started_at) * 1000, 1),
            "image_width": int(image.shape[1]),
            "image_height": int(image.shape[0]),
            "annotated_image_base64": encode_jpeg(annotated),
            "devices": self.devices,
        }

    def _recognize_crop(
        self,
        crop: np.ndarray,
    ) -> tuple[str, float, str, float, list[dict[str, object]]]:
        candidates: list[dict[str, object]] = []
        total_started_at = perf_counter()
        for variant_name, variant in plate_preprocessing_variants(crop):
            if variant.ndim == 2:
                variant = cv2.cvtColor(variant, cv2.COLOR_GRAY2BGR)
            variant_started_at = perf_counter()
            outputs = list(
                self._recognizer.predict(
                    input=np.ascontiguousarray(variant, dtype=np.uint8),
                    batch_size=1,
                )
            )
            payload = outputs[0].json.get("res", outputs[0].json) if outputs else {}
            candidates.append(
                {
                    "variant": variant_name,
                    "text": normalize_plate_text(str(payload.get("rec_text", ""))),
                    "confidence": float(payload.get("rec_score", 0.0)),
                    "latency_ms": round((perf_counter() - variant_started_at) * 1000, 1),
                }
            )

        votes = Counter(item["text"] for item in candidates if item["text"])
        for item in candidates:
            item["score"] = round(
                candidate_score(
                    str(item["text"]),
                    float(item["confidence"]),
                    votes[str(item["text"])],
                ),
                4,
            )
        candidates.sort(key=lambda item: float(item["score"]), reverse=True)
        winner = candidates[0]
        public_candidates = [
            {
                "variant": item["variant"],
                "text": item["text"],
                "confidence": round(float(item["confidence"]), 4),
            }
            for item in candidates[:3]
        ]
        return (
            str(winner["text"]),
            float(winner["confidence"]),
            str(winner["variant"]),
            (perf_counter() - total_started_at) * 1000,
            public_candidates,
        )

    @staticmethod
    def _crop_with_margin(
        image: np.ndarray,
        xyxy: list[float],
        margin: float = 0.08,
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        height, width = image.shape[:2]
        x1, y1, x2, y2 = xyxy
        pad_x, pad_y = (x2 - x1) * margin, (y2 - y1) * margin
        left, top = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
        right, bottom = min(width, int(x2 + pad_x)), min(height, int(y2 + pad_y))
        return image[top:bottom, left:right].copy(), (left, top, right, bottom)

    @staticmethod
    def _draw_detection(
        image: np.ndarray,
        coordinates: tuple[int, int, int, int],
        text: str,
        status: str,
        confidence: float,
    ) -> None:
        colors = {"ACCEPT": (35, 150, 75), "REVIEW": (0, 155, 230), "REJECT": (40, 50, 210)}
        color = colors[status]
        x1, y1, x2, y2 = coordinates
        cv2.rectangle(image, (x1, y1), (x2, y2), color, max(2, image.shape[1] // 600))
        label = f"{text or 'UNREADABLE'} | {status} | {confidence:.0%}"
        cv2.putText(
            image,
            label,
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
