"""Package local training artifacts into the runtime model directory."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DETECTOR_SOURCE = PROJECT_ROOT / "ml/experiments/yolo_license_plate_detector/yolo11n_parking_v2/weights/best.pt"
RECOGNIZER_SOURCE = PROJECT_ROOT / "ml/experiments/paddleocr_plate_recognizer/inference_perspective_v3"
TARGET_ROOT = PROJECT_ROOT / "ml/weights/anpr"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    required = [DETECTOR_SOURCE, RECOGNIZER_SOURCE / "inference.yml"]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing model artifacts:\n" + "\n".join(str(path) for path in missing))

    recognizer_target = TARGET_ROOT / "recognizer"
    recognizer_target.mkdir(parents=True, exist_ok=True)
    detector_target = TARGET_ROOT / "license_plate_yolo11n.pt"
    shutil.copy2(DETECTOR_SOURCE, detector_target)
    for source in RECOGNIZER_SOURCE.iterdir():
        if source.is_file():
            shutil.copy2(source, recognizer_target / source.name)

    packaged_files = [detector_target, *sorted(recognizer_target.iterdir())]
    manifest = {
        "model_version": "parking-detector-v2-perspective-recognizer-v3",
        "detector": detector_target.relative_to(PROJECT_ROOT).as_posix(),
        "recognizer": recognizer_target.relative_to(PROJECT_ROOT).as_posix(),
        "files": {
            path.relative_to(PROJECT_ROOT).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in packaged_files
        },
    }
    manifest_path = TARGET_ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Packaged {len(packaged_files)} model files in {TARGET_ROOT}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
