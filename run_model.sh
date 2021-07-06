#!/bin/bash -i

source ~/miniconda3/etc/profile.d/conda.sh

conda activate pangeo

MONTH="OCT"
MONTH_DIR="/home/grivera/GitLab/pstatmodel-runs/RUNS/$MONTH"
SETTINGS_PATH="$MONTH_DIR/settings.json"

BASE_DIR=$(pwd)

cd "$MONTH_DIR"
python create_database.py $SETTINGS_PATH

cd "$BASE_DIR/src"
sbatch --export=ALL,SETTINGS_PATH=$SETTINGS_PATH --output=$MONTH_DIR/modelrun-log.txt --job-name="model$MONTH" run_model.sbatch

cd "$BASE_DIR"
papermill notebooks/plotting_template.ipynb $MONTH_DIR/plotting_$MONTH.ipynb $SETTINGS_PATH
