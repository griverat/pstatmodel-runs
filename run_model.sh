#!/bin/bash -i

source ~/miniconda3/etc/profile.d/conda.sh

conda activate pstatmodel

DATE="2022/07"
MONTH=$(date -d $DATE/01 +%Y.%m)
MONTH_PATH=$(date -d $DATE/01 +%Y/%m.%^b)
MONTH_DIR="/home/grivera/GitLab/pstatmodel-runs/RUNS/$MONTH_PATH"
LOGS="$MONTH_DIR/logs"
SETTINGS_PATH="$MONTH_DIR/settings.json"

mkdir -p $LOGS

BASE_DIR=$(pwd)

cd "$BASE_DIR/src"
python create_database.py $SETTINGS_PATH
sbatch -W --export=ALL,SETTINGS_PATH=$SETTINGS_PATH --output=$LOGS/modelrun-log_$(date +%Y-%m-%d_%H:%M).txt --job-name="p$MONTH" run_model.sbatch

wait

cd "$BASE_DIR"
papermill notebooks/plotting_template.ipynb $MONTH_DIR/plotting_$MONTH.ipynb -p settings $SETTINGS_PATH
papermill notebooks/plot_diagnostic_template.ipynb $MONTH_DIR/plot_diagnostic_$MONTH.ipynb -p settings $SETTINGS_PATH
