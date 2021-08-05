#!/bin/bash -i

source ~/miniconda3/etc/profile.d/conda.sh

conda activate pangeo

MONTH="JUL"
MONTH_DIR="/home/grivera/GitLab/pstatmodel-runs/RUNS/$MONTH"
SETTINGS_PATH="$MONTH_DIR/settings.json"

BASE_DIR=$(pwd)

cd "$MONTH_DIR"
python create_database.py $SETTINGS_PATH

cd "$BASE_DIR/src"
sbatch -W --export=ALL,SETTINGS_PATH=$SETTINGS_PATH --output=$MONTH_DIR/fcstrun-log.txt --job-name="fcst$MONTH" run_fcst.sbatch

wait

cd "$BASE_DIR"
papermill notebooks/fcst_plot_template.ipynb $MONTH_DIR/fcst_plot_$MONTH.ipynb -p settings $SETTINGS_PATH