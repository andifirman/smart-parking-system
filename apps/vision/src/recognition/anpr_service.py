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
    return [("original", resized), ("clahe", clahe), ("otsu", otsu), ("adaptive", adaptive)]


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
