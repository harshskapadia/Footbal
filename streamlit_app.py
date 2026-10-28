"""Streamlit dashboard: upload a clip, run the pipeline, explore video, minimap, heatmaps and stats.

Run with:  streamlit run streamlit_app.py
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from src.pipeline import PipelineOptions, run_pipeline
from src.pitch.template import PitchTemplate
from src.utils.cli import RunContext
from src.utils.config import load_config, resolve_path
from src.utils.device import select_device
from src.utils.paths import RunPaths
from src.visualization.minimap import Minimap

st.set_page_config(page_title="Football Analytics", page_icon="⚽", layout="wide")

CFG = load_config()
RAW_DIR = resolve_path("data/raw")
OUTPUT_DIR = resolve_path(CFG["paths"]["output_dir"])
SYNTHETIC_DIR = resolve_path("data/synthetic")
VIDEO_TYPES = ["mp4", "avi", "mov", "mkv"]


# ---------------------------------------------------------------- helpers
def model_status(cfg: dict) -> dict[str, bool]:
    """Which model files are present."""
    return {name: resolve_path(cfg["models"][key]).exists() for name, key in
            [("Player detector", "player_detector"), ("Ball detector", "ball_detector"),
             ("Pitch keypoints", "pitch_detector")]}


def previous_runs() -> list[Path]:
    """Run folders that finished (contain stats.json), newest first."""
    runs = [p.parent for p in OUTPUT_DIR.glob("*/stats.json")]
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)


@st.cache_data(show_spinner=False)
def read_csv(path: str, mtime: float) -> pd.DataFrame:
    """Cached CSV read (``mtime`` busts the cache when the file changes)."""
    return pd.read_csv(path)


def load_csv(path: Path) -> pd.DataFrame | None:
    return read_csv(str(path), path.stat().st_mtime) if path.exists() else None


def zip_outputs(root: Path) -> bytes:
    """All CSV/JSON/PNG outputs of a run as one zip."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(root.iterdir()):
            if f.suffix in {".csv", ".json", ".png"}:
                zf.write(f, f.name)
    return buf.getvalue()


def team_colours(root: Path) -> dict[int, tuple[int, int, int]]:
    path = root / "team_colours.json"
    if not path.exists():
        return {0: (60, 60, 220), 1: (220, 160, 60)}
    return {int(k): tuple(v["bgr"]) for k, v in json.loads(path.read_text()).items()}


# ---------------------------------------------------------------- sidebar
st.sidebar.title("⚽ Football Analytics")
st.sidebar.caption("Detection · tracking · teams · pitch mapping · stats")

with st.sidebar.expander("Models", expanded=True):
    status = model_status(CFG)
    for name, ok in status.items():
        st.write(("✅ " if ok else "❌ ") + name)
    if not status["Player detector"]:
        st.caption("No football detector: using COCO `yolo11n` (no referee/goalkeeper classes).")
    if not status["Pitch keypoints"]:
        st.caption("No pitch model: speed, distance, heatmaps and minimap need it. "
                   "Run `bash scripts/download_assets.sh`.")
    st.write(f"Device: `{select_device(CFG.get('device', 'auto'))}`")

st.sidebar.subheader("Settings")
max_frames = st.sidebar.number_input("Max frames (0 = whole clip)", min_value=0, value=250, step=50,
                                     help="Detection is slow on CPU; start with a few hundred frames.")
use_slicing = st.sidebar.toggle("Sliced ball inference", value=CFG["ball"]["use_slicing"],
                                help="Better small-ball recall, roughly 4x slower.")
det_conf = st.sidebar.slider("Detection confidence", 0.1, 0.9, float(CFG["detection"]["conf"]), 0.05)

# ---------------------------------------------------------------- input
st.title("Match analysis")
source = st.radio("Input", ["Upload a clip", "Clip in data/raw", "Previous results"], horizontal=True)

run_root: Path | None = st.session_state.get("run_root")
clip: Path | None = None
options = PipelineOptions()

if source == "Upload a clip":
    upload = st.file_uploader("Broadcast clip", type=VIDEO_TYPES)
    if upload is not None:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        clip = RAW_DIR / Path(upload.name).name
        clip.write_bytes(upload.getbuffer())
elif source == "Clip in data/raw":
    clips = sorted(p for p in RAW_DIR.glob("*") if p.suffix.lower().lstrip(".") in VIDEO_TYPES)
    if clips:
        clip = st.selectbox("Clip", clips, format_func=lambda p: p.name)
        synth = SYNTHETIC_DIR / clip.stem
        if (synth / "detections.csv").exists():
            st.info("Synthetic clip: using the simulator's detections and keypoints instead of the models.")
            options = PipelineOptions(synth / "detections.csv", synth / "keypoints.csv")
    else:
        st.warning("No clips in data/raw. Download samples (README) or run `python -m src.simulation`.")
else:
    runs = previous_runs()
    if runs:
        run_root = st.selectbox("Run", runs, format_func=lambda p: p.name)
        st.session_state["run_root"] = run_root
    else:
        st.info("No finished runs in outputs/ yet.")

if clip is not None and st.button("Run analysis", type="primary"):
    cfg = load_config()
    cfg["ball"]["use_slicing"] = use_slicing
    cfg["detection"]["conf"] = det_conf
    ctx = RunContext(cfg, clip, RunPaths.for_clip(OUTPUT_DIR, clip), int(max_frames) or None)
    bar = st.progress(0.0, text="Starting…")
    try:
        def report(fraction: float, stage: str) -> None:
            bar.progress(fraction, text=f"{stage}… {fraction:.0%}")

        result = run_pipeline(ctx, options, report)
    except Exception as e:  # noqa: BLE001 - show any failure in the UI
        bar.empty()
        st.error(f"Pipeline failed: {e}")
        st.stop()
    bar.progress(1.0, text="Done")
    st.session_state["run_root"] = run_root = result.root

# ---------------------------------------------------------------- results
if run_root is None or not (run_root / "stats.json").exists():
    st.stop()

paths = RunPaths(run_root)
stats = json.loads(paths.stats_json.read_text())
st.header(f"Results: {stats['clip']}")
st.caption(f"{stats['frames']} frames · {stats['duration_s']} s · {stats['fps']:.0f} fps")
kits = stats.get("team_colours_hex", {})
if kits:
    swatch = "&nbsp;&nbsp;".join(
        f'<span style="display:inline-block;width:12px;height:12px;border-radius:2px;background:{hexc};'
        f'border:1px solid #888;vertical-align:middle"></span>&nbsp;Team {t} kit'
        for t, hexc in kits.items())
    st.markdown(swatch, unsafe_allow_html=True)
for w in stats.get("warnings", []):
    st.warning(w)

poss, passes = stats["possession_pct"], stats["passes"]
cols = st.columns(4)
cols[0].metric("Possession: team 0", f"{poss['0']}%")
cols[1].metric("Possession: team 1", f"{poss['1']}%")
cols[2].metric("Passes completed: team 0", f"{passes['0']['completed']}/{passes['0']['attempted']}",
               help=f"Accuracy {passes['0']['accuracy_pct']}%")
cols[3].metric("Passes completed: team 1", f"{passes['1']['completed']}/{passes['1']['attempted']}",
               help=f"Accuracy {passes['1']['accuracy_pct']}%")
if "teams" in stats:
    cols = st.columns(4)
    for i, team in enumerate(("0", "1")):
        cols[2 * i].metric(f"Distance: team {team}", f"{stats['teams'][team]['total_distance_m'] / 1000:.2f} km")
        cols[2 * i + 1].metric(f"Top speed: team {team}", f"{stats['teams'][team]['top_speed_kmh']} km/h")

tab_video, tab_map, tab_net, tab_tables, tab_dl = st.tabs(
    ["Video", "Minimap & heatmaps", "Passing networks", "Tables", "Downloads"])

with tab_video:
    video = paths.web_video if paths.web_video.exists() else paths.annotated_video
    if video.exists():
        st.video(str(video))
        if video == paths.annotated_video:
            st.caption("H.264 copy missing: the browser may not play mp4v. Download it from the Downloads tab.")
    else:
        st.info("No annotated video for this run.")

with tab_map:
    positions = load_csv(paths.positions)
    ball_pos = load_csv(paths.ball_positions)
    possession = load_csv(paths.possession)
    if positions is not None and ball_pos is not None and positions["x_m"].notna().any():
        frame = st.slider("Frame", 0, int(stats["frames"]) - 1, 0)
        minimap = Minimap(PitchTemplate.from_config(CFG), team_colours(run_root), scale=8.0)
        people = positions[positions["frame"] == frame]
        ball_xy = ball_pos.loc[ball_pos["frame"] == frame, ["x_m", "y_m"]].to_numpy(float)
        owner = int(possession.loc[possession["frame"] == frame, "owner_id"].iloc[0]) if possession is not None else -1
        img = minimap.render(people, ball_xy[0] if len(ball_xy) else None, owner)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"Top-down view, frame {frame} "
                 "(gold ring = player on the ball)", width=760)
    else:
        st.info("The minimap needs a pitch mapping (pitch keypoint model).")
    heat = [paths.heatmap(t) for t in (0, 1) if paths.heatmap(t).exists()]
    if heat:
        c = st.columns(len(heat))
        for col, img_path in zip(c, heat, strict=True):
            col.image(str(img_path), width="stretch")

with tab_net:
    nets = [paths.passing_network(t) for t in (0, 1) if paths.passing_network(t).exists()]
    if nets:
        c = st.columns(len(nets))
        for col, img_path in zip(c, nets, strict=True):
            col.image(str(img_path), width="stretch")
    else:
        st.info("Passing networks need a pitch mapping.")
    pass_df = load_csv(paths.passes)
    if pass_df is not None:
        st.dataframe(pass_df, width="stretch", hide_index=True)

with tab_tables:
    team_df = load_csv(run_root / "team_summary.csv")
    if team_df is not None:
        st.subheader("Teams")
        st.dataframe(team_df, width="stretch", hide_index=True)
    player_df = load_csv(paths.player_stats)
    if player_df is not None:
        st.subheader("Players")
        st.dataframe(player_df, width="stretch", hide_index=True,
                     column_config={"distance_m": st.column_config.NumberColumn("Distance (m)", format="%.0f"),
                                    "max_speed_kmh": st.column_config.NumberColumn("Top speed (km/h)"),
                                    "avg_speed_kmh": st.column_config.NumberColumn("Avg speed (km/h)")})
    teams_df = load_csv(paths.teams)
    if teams_df is not None:
        st.subheader("Track roles and teams")
        st.dataframe(teams_df, width="stretch", hide_index=True)

with tab_dl:
    st.download_button("All stats (zip of CSV/JSON/PNG)", zip_outputs(run_root), f"{run_root.name}_stats.zip",
                       "application/zip")
    st.download_button("stats.json", paths.stats_json.read_bytes(), "stats.json", "application/json")
    if paths.annotated_video.exists():
        st.download_button("Annotated video (mp4)", paths.annotated_video.read_bytes(), "annotated.mp4", "video/mp4")
    st.caption(f"All files are also in `{run_root}`.")
