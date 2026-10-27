# Football Analytics (Computer Vision)

Turn a broadcast football clip into an annotated video and match stats: player
detection and tracking, team assignment, ball possession, speed and distance,
heatmaps and a 2D minimap.

> Work in progress. The project is built in stages, and this README grows as each
> stage lands.

| Stage | Status |
|---|---|
| 1. Setup (structure, config, env) | done |
| 2. Detection (YOLO, sliced ball inference, fine-tuning) | in progress |
| 3. Tracking (ByteTrack, ball interpolation) | todo |
| 4. Team assignment (shirt colour K-means) | todo |
| 5. Pitch mapping (keypoints, homography) | todo |
| 6. Analytics (possession, speed, heatmaps, passes, minimap) | todo |
| 7. Output (video, CSV, JSON) | todo |
| 8. Streamlit dashboard | todo |
| 9. Tests, typing, CLI | todo |
| 10. Documentation | todo |

## Repository layout

```
config.yaml            paths, model locations and thresholds
configs/               training configs
src/
  utils/               config, device selection, video I/O
  detection/           detector, sliced ball inference, dataset download, training
  detect.py            stage 2 CLI: detection on a clip
data/raw/              input clips (not committed)
data/datasets/         Roboflow training datasets (not committed)
models/                model weights (not committed)
outputs/               generated videos and stats (not committed)
notebooks/             exploration notebooks
scripts/               asset download helpers
tests/                 pytest suite
```

## Setup

Requires Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q
```

On Linux, `pip install torch` from PyPI ships the CUDA build, which also runs on
CPU. On a CPU-only machine you can save about 2 GB by installing the CPU wheel first:
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.

The device is picked automatically (`device: auto` in `config.yaml`): CUDA,
then Apple MPS, then CPU.

## Data and models (manual download)

Nothing here requires you to record footage. Everything comes from public sources.

### 1. Sample clips and pretrained football models (recommended)

Roboflow's open-source [sports](https://github.com/roboflow/sports) project
publishes YOLOv8 models trained on football data, plus five 30-second broadcast
clips from the Kaggle
[DFL - Bundesliga Data Shootout](https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout).
They are hosted on Google Drive:

```bash
pip install gdown
bash scripts/download_assets.sh
```

This creates:

| File | Purpose |
|---|---|
| `models/football-player-detection.pt` | ball / goalkeeper / player / referee detector |
| `models/football-ball-detection.pt` | dedicated ball detector |
| `models/football-pitch-detection.pt` | 32 pitch keypoints (stage 5) |
| `data/raw/{0bfacc_0,2e57b9_0,08fd33_0,573e61_0,121364_0}.mp4` | sample clips |

If the Google Drive downloads fail, you can use any clip from the Kaggle competition
instead: accept its rules, download `clips/*.mp4` and put a clip in `data/raw/`.

### 2. Training datasets (only if you want to fine-tune)

Roboflow Universe datasets (free account and API key needed:
https://app.roboflow.com/settings/api):

| Dataset | Version | Task |
|---|---|---|
| [football-players-detection-3zvbc](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc) | 10 | detection, 4 classes |
| [football-ball-detection-rejhg](https://universe.roboflow.com/roboflow-jvuqo/football-ball-detection-rejhg) | 2 | detection, ball only |
| [football-field-detection-f07vi](https://universe.roboflow.com/roboflow-jvuqo/football-field-detection-f07vi) | 12 | pose, 32 keypoints |

```bash
pip install -r requirements-train.txt
export ROBOFLOW_API_KEY=your_key
python -m src.detection.download_dataset --dataset players     # -> data/datasets/football-players-detection-10/
python -m src.detection.train --config configs/train_detector.yaml
```

The trained weights are copied to `models/football-player-detection.pt`.

### 3. Without any downloads

If no football model is present, the pipeline falls back to the COCO
`yolo11n.pt` model, which Ultralytics downloads automatically from GitHub. It
detects `person` (mapped to *player*) and `sports ball` (mapped to *ball*), but it
cannot tell referees and goalkeepers apart. Use it for smoke tests only.

## Stage 2: detection

```bash
python -m src.detect --input data/raw/0bfacc_0.mp4 --max-frames 150
```

Outputs:
- `outputs/<clip>_detections.mp4`: annotated video
- `outputs/<clip>_detections.csv`: one row per box
  (`frame, class_id, class_name, confidence, x1, y1, x2, y2`)

The ball is often only 5–15 px wide in broadcast footage. With `ball.use_slicing`
on, each frame is cut into overlapping 640×640 tiles (`supervision.InferenceSlicer`).
The ball model runs on every tile, the results are merged with NMS, and the most
confident ball is kept.
