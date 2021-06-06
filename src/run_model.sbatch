#!/bin/bash -l
#SBATCH --partition mpi_long1
#SBATCH --ntasks 1
#SBATCH --time 5-00:00:00
#SBATCH --mem-per-cpu 120000
#SBATCH --job-name modelMAY
#SBATCH --output modelrun-log-%J.txt

source ~/miniconda3/etc/profile.d/conda.sh

conda activate pangeo

ipnip=$(hostname -i)
user='grivera'
host='10.10.90.12'

echo -e "
    Copy/Paste this in your local terminal to ssh tunnel with remote
    dask dashboard
    -----------------------------------------------------------------
    ssh -N -L 8787:$ipnip:8787 $user@$host
    -----------------------------------------------------------------
    
    From outside the network.
    Using an ssh key with host and port details in .ssh/config
    -----------------------------------------------------------------
    ssh -N -L 8787:$ipnip:8787 $user
    -----------------------------------------------------------------
    "

echo "Using settings from :" $SETTINGS_PATH

python run_model.py $SETTINGS_PATH
python post_run.py $SETTINGS_PATH
python make_forecast.py $SETTINGS_PATH