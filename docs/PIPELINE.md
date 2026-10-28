# Pipeline details

How each stage works, what it writes, and the design decisions behind it. Every
stage reads and writes files in `outputs/<clip>/`, so each one can be re-run on its
own. See the [README](../README.md) for setup and the overview.

## Shot segmentation (before detection)

Edited footage (highlights, full broadcasts) switches between cameras, so the clip
is first split into shots (`src/shots/segmentation.py`, settings under `shots:` in
`config.yaml`):

1. **Scan:** for every frame (downscaled to 160×90), a hue/saturation histogram and
   the share of grass pixels.
2. **Cuts:** the Bhattacharyya distance between consecutive histograms must be above
   0.35 **and** 4× the local median, so steady fast pans don't count. Shots shorter
   than 5 frames (flashes, wipes, encoder glitches) are merged into a neighbour.
3. **Wide or not:** median grass ratio ≥ 0.35 and length ≥ 1 s mark a candidate.
   After detection, a candidate is rejected if its median player box is taller than
   22% of the frame (a close-up on grass) or it has no players.
4. **Everything downstream respects shots:** detection and keypoints run only on
   candidates; the tracker restarts at each cut, with new IDs continuing upwards;
   ball interpolation, homography smoothing and possession stay within a shot; and a
   pass needs both players in the same shot.

On a test reel (wide shot → close-up → non-pitch shot → wide shot from a different
camera position), all three cuts were found, only the two wide shots were analysed,
and no track, homography or pass crossed a cut. `tests/test_shots.py` checks this.

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

## Stage 6: analytics

```bash
python -m src.analyze --input data/raw/0bfacc_0.mp4
```

| Output | What it is |
|---|---|
| `possession.csv` | per frame: `owner_id`, `owner_team`, distance to the ball, `team_in_possession` |
| `passes.csv` | `frame_start, frame_end, from_id, to_id, team, to_team, completed, distance_m` |
| `player_stats.csv` | per track: distance (m), average and top speed (km/h), minutes tracked |
| `speeds.csv` | per frame and track speed, used for the video overlay |
| `heatmap_team{0,1}.png` | where each team spent its time (shared colour scale) |
| `passing_network_team{0,1}.png` | players at their average position, lines weighted by passes between them |
| `stats.json` | headline summary |

- **Possession:** the nearest player (or goalkeeper) within 2 m of the ball controls
  it. Spells shorter than 0.4 s are ignored: a pass at 16 m/s stays within 2 m of
  a player it flies past for about 0.25 s. While the ball travels, the last team in
  control keeps possession.
- **Passes:** a change of owner between consecutive control spells. To a teammate it
  is completed; to an opponent it is a turnover. Changes after more than 4 s of loose
  ball, and "passes" shorter than 3 m (usually one player split into two IDs), are
  ignored.
- **Speed and distance:** positions are smoothed with a centred 0.4 s window, then
  measured frame to frame. Steps across gaps in a track, and readings above 40 km/h,
  are discarded. Top speed is each player's 95th-percentile reading, so a single
  noisy frame can't set it. Smoothing matters: on the synthetic match, raw positions
  overestimate distance by 55%; the 0.4 s window brings that to +1.6%.
- **Minimap:** a top-down pitch with players in their kit colours, the player on the
  ball ringed in gold, and the ball. It is drawn onto the video in stage 7.
- **Without a pitch mapping**, possession and passes are measured in pixels, and the
  metric outputs (speed, distance, heatmaps, networks) are skipped with a warning.

Charts use one sequential blue ramp for magnitude, and fixed, colour-blind-checked
team colours (team 0 blue `#2a78d6`, team 1 orange `#eb6834`). Kit colours aren't
used in charts because a white or green kit would disappear against the background.

## Stage 7: full pipeline and outputs

```bash
python -m src.run --input data/raw/0bfacc_0.mp4              # everything, written to outputs/0bfacc_0/
python -m src.run --input clip.mp4 --max-frames 250 --device cpu
python -m src.run --input data/raw/synthetic_match.mp4 \
    --detections data/synthetic/synthetic_match/detections.csv \
    --keypoints  data/synthetic/synthetic_match/keypoints.csv
```

- `annotated.mp4`: players in their team's kit colour with ID and live speed, a gold
  marker on the player in possession, the ball (orange when interpolated), a running
  possession bar, and the 2D minimap.
- `annotated_h264.mp4`: the same video re-encoded to H.264 with the ffmpeg bundled in
  `imageio-ffmpeg`, so browsers and the dashboard can play it. OpenCV's pip build
  can only write `mp4v`.
- `stats.json`: possession %, passes per team, distance and top speed per team, top
  5 players by distance, team kit colours, and any warnings.
- `team_summary.csv`, `player_stats.csv`, `passes.csv`, `possession.csv`, plus every
  intermediate CSV from stages 2–5.

If the pitch keypoint model is missing, the run still finishes. Possession and passes
are measured in pixels, the metric outputs are skipped, and the reason is shown in the
warnings.

## Stage 8: dashboard

```bash
streamlit run streamlit_app.py
```

- **Input:** upload a clip, pick one from `data/raw/`, or reopen a finished run from
  `outputs/`. Synthetic clips automatically use the simulator's detections and
  keypoints.
- **Sidebar:** which models were found, the compute device, a frame limit (CPU runs
  are slow), sliced ball inference on or off, and detection confidence.
- **Results:** headline numbers (possession, passes, distance, top speed) with kit
  colour swatches, then tabs for the annotated video (H.264), the minimap with a frame
  slider, heatmaps, passing networks, tables (teams, players, passes, track roles), and
  downloads (zip of all stats, `stats.json`, video).
- A progress bar covers the whole pipeline. Warnings, such as a missing pitch model,
  are shown in the page.

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

Score a run against the ground truth with `python -m src.simulation.evaluate`. It
writes `evaluation.json` with tracking fragmentation, team accuracy, position error,
possession agreement, pass precision and recall, and distance error.
