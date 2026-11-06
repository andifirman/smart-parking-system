# Smart Parking System

## What This Application Does

Smart Parking System is an automated parking access project that combines IP
camera capture, Automatic Number Plate Recognition (ANPR), access validation,
and a physical barrier controller.

The intended flow is:

```text
IP Camera / Webcam
        |
        v
YOLO license-plate detector
        |
        v
Plate crop and image preprocessing
        |
        v
OCR / plate-text recognizer
        |
        v
Access decision
        |
        v
ESP32 / Arduino barrier controller
```

ANPR is the Computer Vision capability inside the product, not the name of the
entire application. The current development phase focuses on detecting an
Indonesian license plate in an image and converting the detected crop into a
license-plate string.

## Repository Structure

```text
005. Smart Parking System/
|-- apps/
|   |-- api/             # Backend API
|   |-- vision/          # Production Computer Vision modules
|   `-- web/             # Operator dashboard
|-- data/                # Local datasets and captures (ignored by Git)
|-- docs/                # Product and engineering documentation
|-- firmware/            # ESP32 / Arduino barrier-controller firmware
|-- infrastructure/      # Deployment and operational resources
|-- ml/
|   |-- experiments/     # Generated training outputs (ignored by Git)
|   |-- notebooks/       # Research and model-development notebooks
|   |-- training/        # Future reusable training code
|   `-- weights/         # Local model weights (ignored by Git)
`-- tests/
```

The notebooks live in `ml/notebooks/`, not at the repository root.

## Current Notebooks

### `01_license_plate_detection_and_ocr_baseline.ipynb`

Establishes the OCR baseline using the plate crops and Pascal VOC annotations
from the Indonesian Plate Number dataset. It reports exact-match accuracy,
character similarity, OCR confidence, and processing latency.

### `02_yolo_license_plate_detector_training.ipynb`

Audits and converts Pascal VOC annotations to Ultralytics YOLO format, creates
deterministic train/validation/test splits, trains a YOLO11n plate detector,
evaluates its mAP, and runs the end-to-end flow:

```text
full image -> YOLO bounding box -> plate crop -> EasyOCR -> normalized text
```

## Prerequisites

- Git
- Python 3.11 or 3.12
- Visual Studio Code
- The VS Code **Python** and **Jupyter** extensions
- Internet access for the initial dependency, dataset, and model downloads
- At least 5 GB of free disk space

An NVIDIA GPU is optional. OCR and inference work on CPU, but YOLO training can
be considerably slower. Check the active PyTorch device in the first notebook
cells before starting a long training run.

### Parking-domain detector training

The web application uses a YOLO detector fine-tuned for parking-camera images.
Training is isolated in `.venv_yolo` so its CUDA-enabled PyTorch installation
does not conflict with PaddleOCR in the main `.venv`.

```powershell
py -3.12 -m venv .venv_yolo
.\.venv_yolo\Scripts\python.exe -m pip install ultralytics PyYAML
.\.venv_yolo\Scripts\python.exe -m pip install "torch==2.14.1+cu126" "torchvision==0.29.1+cu126" --index-url https://download.pytorch.org/whl/cu126
.\.venv_yolo\Scripts\python.exe ml\training\train_parking_plate_detector.py
.\.venv\Scripts\python.exe infrastructure\scripts\package_anpr_models.py
```

The training script expects the Roboflow parking dataset at
`TEMPORARY/Test V2/`, writes the best checkpoint to
`ml/experiments/yolo_license_plate_detector/yolo11n_parking_v2/weights/best.pt`,
and evaluates it against the provided test split. Generated datasets, runs,
and packaged weights remain ignored by Git.

For oblique plates, build the polygon-only dataset and fine-tune the YOLO11n
segmentation model in the isolated YOLO environment:

```powershell
.\.venv_yolo\Scripts\python.exe ml\training\build_plate_segmentation_dataset.py
.\.venv_yolo\Scripts\python.exe ml\training\train_plate_segmentation.py
.\.venv\Scripts\python.exe infrastructure\scripts\package_anpr_models.py
```

The runtime uses the segmentation mask to rectify the plate perspective before
OCR. If a valid quadrilateral cannot be recovered, it falls back to the regular
bounding-box crop. Training artifacts are written to
`ml/experiments/yolo_plate_segmentation/yolo11n_plate_seg_v1/`.

The OCR recognizer can be adapted to oblique parking-camera views with a
balanced mix of original and deterministic perspective-augmented crops:

```powershell
.\.venv\Scripts\python.exe ml\training\build_perspective_ocr_dataset.py
.\.venv\Scripts\python.exe ml\training\paddleocr_runner.py TEMPORARY\PaddleOCR\tools\train.py -c ml\experiments\paddleocr_plate_recognizer\plate_recognizer_perspective_v3.yml
.\.venv\Scripts\python.exe ml\training\evaluate_plate_recognizers.py
```

Validation and test images remain unmodified. The runtime packager uses the
exported `inference_perspective_v3` recognizer.

## Windows Setup

Open PowerShell and run:

```powershell
git clone https://github.com/andifirman/smart-parking-system.git
cd smart-parking-system

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name smart-parking-system --display-name "Python (.venv) - Smart Parking System"
```

If PowerShell blocks virtual-environment activation, allow scripts only for the
current terminal session, then activate it again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Verify the environment:

```powershell
python -c "import cv2, easyocr, torch, ultralytics; print('Environment OK'); print('CUDA:', torch.cuda.is_available())"
```

## macOS Setup

Open Terminal and run:

```bash
git clone https://github.com/andifirman/smart-parking-system.git
cd smart-parking-system

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name smart-parking-system --display-name "Python (.venv) - Smart Parking System"
```

Verify the environment:

```bash
python -c "import cv2, easyocr, torch, ultralytics; print('Environment OK'); print('MPS:', torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False)"
```

The notebooks currently fall back to CPU when CUDA is unavailable. This works
on both Intel and Apple Silicon Macs, although training speed depends heavily on
the machine.

## Download the Dataset

Activate the virtual environment and run this command from the repository root.
It works in both PowerShell and macOS Terminal:

```bash
python -c "from pathlib import Path; import kagglehub; target=Path('data/raw/indonesian-plate-number'); target.mkdir(parents=True, exist_ok=True); print(kagglehub.dataset_download('imamdigmi/indonesian-plate-number', output_dir=str(target)))"
```

The expected dataset location is:

```text
data/raw/indonesian-plate-number/
`-- plate_number_recognition_datasets_v1/
    |-- annotations/
    |   |-- train/
    |   `-- test/
    `-- images/
        |-- train/
        `-- test/
```

The dataset is intentionally ignored by Git.

## Download the YOLO11n Base Model

Run this cross-platform command from the repository root:

```bash
python -c "from pathlib import Path; from ultralytics import YOLO; YOLO('yolo11n.pt'); target=Path('ml/weights'); target.mkdir(parents=True, exist_ok=True); Path('yolo11n.pt').replace(target/'yolo11n.pt'); print(target/'yolo11n.pt')"
```

Confirm that this file exists:

```text
ml/weights/yolo11n.pt
```

Model weights and generated training runs are intentionally ignored by Git.

## Run the Notebooks in VS Code

1. Open the cloned repository in VS Code.
2. Open `ml/notebooks/01_license_plate_detection_and_ocr_baseline.ipynb`.
3. Select the kernel **Python (.venv) - Smart Parking System**.
4. Run the cells from top to bottom.
5. Open `ml/notebooks/02_yolo_license_plate_detector_training.ipynb` using the
   same kernel.
6. Run its cells from top to bottom.

EasyOCR downloads its language-model weights on the first run. This happens
only once for the local user profile.

## Train and Test the YOLO Detector

On a fresh clone, open the configuration cell in the second notebook and set:

```python
RUN_TRAINING = True
RUN_EXPORT = False
```

For a short pipeline check, begin with:

```python
EPOCHS = 5
BATCH_SIZE = 4
```

After the full flow succeeds, use the baseline training configuration:

```python
EPOCHS = 50
BATCH_SIZE = 8
```

The best checkpoint is generated at:

```text
ml/experiments/yolo_license_plate_detector/
`-- yolo11n_plate_v1/
    `-- weights/
        |-- best.pt
        `-- last.pt
```

After `best.pt` exists, change the configuration back to:

```python
RUN_TRAINING = False
```

Run the remaining cells to:

- calculate validation and test mAP;
- inspect held-out predictions;
- detect and crop plates from full images;
- read each crop with OCR;
- upload a custom vehicle image for end-to-end testing; and
- optionally export the trained detector to ONNX.

## Generated Files and Git

The following files stay local and must not be committed:

- `.venv/`
- `.env`
- downloaded datasets under `data/`
- generated YOLO datasets under `data/processed/`
- model weights under `ml/weights/`
- training runs under `ml/experiments/`
- temporary research datasets under `TEMPORARY/`

Only source code, documentation, configuration, empty-directory placeholders,
and notebooks without execution output should be committed.

## Development Status

## Run the Local ANPR Web App

Package the locally trained detector and recognizer once:

```powershell
python infrastructure/scripts/package_anpr_models.py
```

The runtime artifacts are copied to `ml/weights/anpr/`. This directory is
ignored by Git and must be distributed through approved model storage for
other machines.

Start the local web application from the repository root:

```powershell
python -m uvicorn apps.api.app.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. The upload is decoded and processed in memory;
the application does not save the raw image. The API loads YOLO and PaddleOCR
once, then reuses both model instances for subsequent requests.

Runtime paths and thresholds can be changed through the variables documented
in `.env.example`. The current `ACCEPT` threshold is provisional and must be
calibrated against validation data before controlling a physical barrier.

Current phase: **Phase 1 - image and webcam-based license-plate detection using
YOLO, followed by OCR recognition**.

Next priorities are improving plate-text recognition, collecting images from the
planned IP Camera position, and moving validated notebook logic into reusable
production modules under `apps/vision/`.
