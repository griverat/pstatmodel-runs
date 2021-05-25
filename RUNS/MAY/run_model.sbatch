#!/bin/bash -l
#SBATCH --partition mpi_long1
#SBATCH --ntasks 1
#SBATCH --time 5-00:00:00
#SBATCH --mem-per-cpu 120000
#SBATCH --job-name modelMAY
#SBATCH --output modelrun-log-%J.txt

conda activate pangeo

echo $(hostname -i)

python run_model.py