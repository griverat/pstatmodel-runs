#!/bin/bash -i

source ~/miniconda3/etc/profile.d/conda.sh

conda activate pangeo

DATE="2021/09"
MONTH=$(date -d $DATE/01 +%Y.%m)
MONTH_PATH=$(date -d $DATE/01 +%Y/%m.%^b)
MONTH_DIR="/home/grivera/GitLab/pstatmodel-runs/RUNS/$MONTH_PATH"
SETTINGS_PATH="$MONTH_DIR/settings.json"

BASE_DIR=$(pwd)

cd "$BASE_DIR/src"
python create_database.py $SETTINGS_PATH
sbatch -W --export=ALL,SETTINGS_PATH=$SETTINGS_PATH --output=$MONTH_DIR/modelrun-log.txt --job-name="p$MONTH" run_model.sbatch

wait

cd "$BASE_DIR"
papermill notebooks/plotting_template.ipynb $MONTH_DIR/plotting_$MONTH.ipynb -p settings $SETTINGS_PATH
papermill notebooks/plot_diagnostic_template.ipynb $MONTH_DIR/plot_diagnostic_$MONTH.ipynb -p settings $SETTINGS_PATH
