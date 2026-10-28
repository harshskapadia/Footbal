# Football Analytics (Computer Vision)

Turn a broadcast football clip into an annotated video and match stats: player
detection and tracking, team assignment, ball possession, speed and distance,
heatmaps and a 2D minimap.

> Work in progress. The project is built in stages, and this README grows as each
> stage lands.

| Stage | Status |
|---|---|
| 1. Setup (structure, config, env) | done |
| 2. Detection (YOLO, sliced ball inference, fine-tuning) | done |
| 3. Tracking (ByteTrack, ball interpolation) | done |
| 4. Team assignment (shirt colour K-means) | done |
| 5. Pitch mapping (keypoints, homography) | done |
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
  tracking/            ByteTrack wrapper, ball outlier removal and interpolation
  team_assignment/     shirt crops, Lab colour features, K-means, referee/goalkeeper rules
  pitch/               pitch template (32 keypoints, 105x68 m) and top-down drawing
  simulation/          synthetic match generator + evaluation against ground truth
  visualization/       frame annotators
  detect.py            stage 2 CLI: detection on a clip
  track.py             stage 3 CLI: tracking on a clip
  assign_teams.py      stage 4 CLI: team assignment
  map_pitch.py         stage 5 CLI: pixels -> pitch metres
data/raw/              input clips (not committed)
data/datasets/         Roboflow training datasets (not committed)
models/                model weights (not committed)
outputs/               generated videos and stats (not committed)
notebooks/             exploration notebooks
scripts/               asset download and smoke-clip helpers
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

The trained weights are copied to `models/football-player-detection.pt`. For the
pitch keypoint model, download `--dataset pitch` and train with
`configs/train_pitch.yaml`. Mosaic augmentation and horizontal flips are turned off
there because they would scramble the keypoint labels.

### 3. Without any downloads

If no football model is present, the pipeline falls back to the COCO
`yolo11n.pt` model, which Ultralytics downloads automatically from GitHub. It
detects `person` (mapped to *player*) and `sports ball` (mapped to *ball*), but it
cannot tell referees and goalkeepers apart. Use it for smoke tests only.

## Stage 2: detection

```bash
python -m src.detect --input data/raw/0bfacc_0.mp4 --max-frames 150
```

Outputs (every stage writes into `outputs/<clip>/`):
- `outputs/<clip>/detections.mp4`: annotated video
- `outputs/<clip>/detections.csv`: one row per box
  (`frame, class_id, class_name, confidence, x1, y1, x2, y2`)

For a quick check without football data, build a short clip from the sample images
that ship with Ultralytics. These are not football frames; the clip only proves the
pipeline runs:

```bash
python scripts/make_smoke_clip.py --out data/raw/smoke.mp4
python -m src.detect --input data/raw/smoke.mp4
```

Measured on a 4-core CPU with the `yolo11n` fallback and slicing on: about 1.7 fps
at 1280×720. Expect much slower speeds with the YOLOv8x football models on CPU, so use
`--max-frames` while testing, or a GPU.

The ball is often only 5–15 px wide in broadcast footage. With `ball.use_slicing`
on, each frame is cut into overlapping 640×640 tiles (`supervision.InferenceSlicer`).
The ball model runs on every tile, the results are merged with NMS, and the most
confident ball is kept.

## Stage 3: tracking

```bash
python -m src.track --input data/raw/0bfacc_0.mp4
```

This reuses `outputs/0bfacc_0/detections.csv` if stage 2 has already run, otherwise
it runs detection first.

Outputs:
- `outputs/<clip>/tracks.mp4`: people with stable IDs (`#7`, `GK #3`, `REF #12`),
  ball in yellow when detected and orange when interpolated
- `outputs/<clip>/tracks.csv`: stage 2 columns plus `tracker_id`
- `outputs/<clip>/ball.csv`: one row per frame (`frame, x1, y1, x2, y2, confidence, interpolated`)

How it works:
- **People** are tracked with ByteTrack from Roboflow's
  [`trackers`](https://github.com/roboflow/trackers) package. ByteTrack moved out of
  `supervision` in later releases, and `trackers` works directly on `sv.Detections`.
  A lost ID is kept alive for `tracking.lost_track_buffer_s` (2 s), so short
  occlusions don't create new IDs. Each track's role (player, goalkeeper or referee)
  is decided by a majority vote over all its frames, which removes per-frame
  class flicker.
- **The ball** is not tracked by ByteTrack because it is too small, too fast and
  too often missing. Instead it is cleaned in two steps:
  1. **Outlier removal:** a detection that implies a speed above
     `max_speed_px_per_frame` relative to *both* neighbouring detections is
     dropped as a false positive (for example a white boot or a pitch marking).
  2. **Interpolation:** gaps of up to `max_gap_frames` (1 s) are filled linearly.
     Longer gaps and the start and end of the clip are left empty rather than guessed.

## Stage 4: team assignment

```bash
python -m src.assign_teams --input data/raw/0bfacc_0.mp4
```

Outputs: `teams.csv` (one row per track: `tracker_id, role, team_id, n_crops, L, a, b,
colour_distance`), `team_colours.json`, and `team_crops.jpg`, a contact sheet of crops
per group so you can check the clustering by eye.

1. **Crops:** up to 12 boxes per track, spread evenly along it. From each, the
   shirt band is kept: 15–50% of the box height, with 20% of the width trimmed from
   each side.
2. **Colour feature:** the median CIELAB colour of the crop's non-grass pixels
   (grass is detected by its HSV hue). Distances in Lab roughly match how different
   two colours look to a person, and the median ignores numbers and logos.
3. **K-means (k=2)** on the crops of tracks the detector labelled *player*. The fit
   runs twice: crops far from their cluster centre are dropped before the second fit,
   so a stray referee or goalkeeper can't pull a team colour towards it. Teams are
   ordered by lightness (team 0 = darker kit), so labels are the same on every run.
4. **Per track**, the team is a majority vote over its crops.
5. **Referees** get no team. **Goalkeepers** wear their own colours, so they are put
   in the team whose players stand nearest to them on average, compared frame by
   frame so camera pans don't matter. With the COCO fallback model (no
   referee or goalkeeper classes), any "player" whose colour fits neither team
   becomes `other`.

## Testing without real footage: the synthetic match

```bash
python -m src.simulation                     # 20 s clip + noisy detections + keypoints + ground truth
python -m src.simulation.evaluate            # score a run against the ground truth
```

The simulator plays out a 4-4-2 vs 4-4-2 passage of play: formations shift with the
ball, players dribble and pass, and some passes are intercepted. It renders the
play from a broadcast camera that pans and zooms, so the homography changes every
frame. It also writes:
- detector-like boxes with jitter, 3% missed people, 30% missed balls and occasional
  false balls;
- noisy pitch keypoints;
- the true positions, teams, possession and passes.

The rendered figures are too simple for YOLO, so stages 3 onwards run on these
simulated detections. That tests everything after detection against known answers.
It is not evidence of how well the detector does on real footage.

## Stage 5: pitch mapping

```bash
python -m src.map_pitch --input data/raw/0bfacc_0.mp4                  # uses models/football-pitch-detection.pt
python -m src.map_pitch --input clip.mp4 --keypoints my_keypoints.csv  # or pre-computed keypoints
```

Outputs: `positions.csv` (`frame, tracker_id, team_id, role, x_px, y_px, x_m, y_m`),
`ball_positions.csv`, `homographies.csv` (per-frame fit quality and the smoothed
matrix), `keypoints.csv`, and `pitch_overlay.mp4`, which draws the pitch lines back
onto each frame so you can check the mapping by eye.

1. **Keypoints:** a YOLO pose model finds up to 32 pitch landmarks (corners, box
   corners, penalty spots, centre-circle points), in the same order as Roboflow's
   `football-field-detection` dataset. Keypoints below `keypoint_conf` are ignored.
2. **Homography per frame:** `cv2.findHomography` with RANSAC (1 m threshold) maps
   pixels to a 105×68 m template. A frame is rejected if it has fewer than 4 inliers
   or a mean reprojection error above 1.5 m.
3. **Smoothing over time.** Each homography is represented by where a fixed grid of
   12 image points lands on the pitch. Those trajectories are interpolated across
   gaps of up to 1 s, passed through a centred rolling median to remove one-frame
   glitches, then a centred rolling mean to remove jitter. A homography is then refit
   for every frame. The whole clip is processed offline, so the centred windows add
   no lag while the camera pans.
4. **Projection:** each player is projected from the bottom-centre of their box (the
   feet) and the ball from the bottom of its box. Points more than 5 m outside the
   pitch are discarded.

On the synthetic match (panning and zooming camera, 1.5 px keypoint noise, 10%
missing keypoints):

| | People, mean / p95 error | Ball, mean / p95 error |
|---|---|---|
| Per-frame homography only | 0.71 m / 0.60 m | 0.88 m / 0.86 m |
| With temporal smoothing | **0.14 m / 0.32 m** | **0.09 m / 0.19 m** |

(The ball row without smoothing also projected from the box centre, an error fixed
at the same time.)

If no pitch model or keypoints are available, the command stops and explains what to
download.
