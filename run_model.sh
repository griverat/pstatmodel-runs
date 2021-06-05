#!/bin/bash -i

source ~/miniconda3/etc/profile.d/conda.sh

conda activate pangeo

MONTH_DIR="/home/grivera/GitLab/pstatmodel-runs/RUNS/MAY"
SETTINGS_PATH="$MONTH_DIR/settings.json"

BASE_DIR=$(pwd)

cd "$MONTH_DIR"
python create_database.py $SETTINGS_PATH

cd "$BASE_DIR/src"
sbatch --export=SETTINGS_PATH --output=$MONTH_DIR/modelrun-log.txt --job-name="MODEL_SEP" run_model.sbatch
