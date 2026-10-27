#!/usr/bin/env bash
# Download sample clips and pretrained football models published by Roboflow's
# open-source "sports" project (https://github.com/roboflow/sports, examples/soccer/setup.sh).
# The clips are from the Kaggle "DFL - Bundesliga Data Shootout" competition.
#
# Usage:  pip install gdown && bash scripts/download_assets.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/models" "$ROOT/data/raw"

if ! command -v gdown >/dev/null 2>&1; then
  echo "gdown not found. Install it with: pip install gdown" >&2
  exit 1
fi

# Models (YOLOv8x, trained by Roboflow on their Universe football datasets)
gdown -O "$ROOT/models/football-player-detection.pt" "https://drive.google.com/uc?id=17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q"
gdown -O "$ROOT/models/football-ball-detection.pt"   "https://drive.google.com/uc?id=1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V"
gdown -O "$ROOT/models/football-pitch-detection.pt"  "https://drive.google.com/uc?id=1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf"

# Sample broadcast clips (~30 s each)
gdown -O "$ROOT/data/raw/0bfacc_0.mp4" "https://drive.google.com/uc?id=12TqauVZ9tLAv8kWxTTBFWtgt2hNQ4_ZF"
gdown -O "$ROOT/data/raw/2e57b9_0.mp4" "https://drive.google.com/uc?id=19PGw55V8aA6GZu5-Aac5_9mCy3fNxmEf"
gdown -O "$ROOT/data/raw/08fd33_0.mp4" "https://drive.google.com/uc?id=1OG8K6wqUw9t7lp9ms1M48DxRhwTYciK-"
gdown -O "$ROOT/data/raw/573e61_0.mp4" "https://drive.google.com/uc?id=1yYPKuXbHsCxqjA9G-S6aeR2Kcnos8RPU"
gdown -O "$ROOT/data/raw/121364_0.mp4" "https://drive.google.com/uc?id=1vVwjW1dE1drIdd4ZSILfbCGPD4weoNiu"

echo "Done. Models in models/, clips in data/raw/."
