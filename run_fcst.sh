#!/bin/bash -i

source ~/miniconda3/etc/profile.d/conda.sh

conda activate pangeo

DATE="2021/07"
MONTH=$(date -d $DATE/01 +%Y/%m.%^b)
MONTH_DIR="/home/grivera/GitLab/pstatmodel-runs/RUNS/$MONTH"
SETTINGS_PATH="$MONTH_DIR/settings.json"

BASE_DIR=$(pwd)

cd "$BASE_DIR/src"
python create_database.py $SETTINGS_PATH
sbatch -W --export=ALL,SETTINGS_PATH=$SETTINGS_PATH --output=$MONTH_DIR/fcstrun-log.txt --job-name="f$MONTH" run_fcst.sbatch

wait

cd "$BASE_DIR"
papermill notebooks/fcst_plot_template.ipynb $MONTH_DIR/fcst_plot_$MONTH.ipynb -p settings $SETTINGS_PATH