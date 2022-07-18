#!/bin/bash -i

source ~/miniconda3/etc/profile.d/conda.sh

conda activate pstatmodel

DATE="2022/08"
MONTH=$(date -d $DATE/01 +%Y.%m)
MONTH_PATH=$(date -d $DATE/01 +%Y/%m.%^b)
MONTH_DIR="/home/grivera/GitLab/pstatmodel-runs/RUNS/$MONTH_PATH"
SETTINGS_PATH="$MONTH_DIR/settings.json"


BASE_DIR=$(pwd)

cd "$BASE_DIR"
papermill notebooks/reduce_sectors_template.ipynb $MONTH_DIR/reduce_sectors_$MONTH.ipynb -p settings $SETTINGS_PATH
