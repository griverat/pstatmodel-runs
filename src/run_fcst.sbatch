#!/bin/bash -l
#SBATCH --partition mpi_long2
#SBATCH --ntasks 1
#SBATCH --time 5-00:00:00
#SBATCH --mem-per-cpu 120000
#SBATCH --job-name modelMAY
#SBATCH --output modelrun-log-%J.txt

source ~/miniconda3/etc/profile.d/conda.sh

conda activate pangeo
python make_forecast.py $SETTINGS_PATH
