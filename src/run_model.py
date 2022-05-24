import argparse
import os
import pickle

import pandas as pd
from dask import compute, delayed
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from dmelon import utils
from pstatmodel.stepwise import base

import xarray as xr

parser = argparse.ArgumentParser(description="Run the pstatmodel")
parser.add_argument("settings", type=str)
args = parser.parse_args()

settings = args.settings
settings = utils.load_json(settings)

MONTH = settings["MONTH"]
DATA_DIR = settings["DATA_DIR"]
MONTH_DIR = os.path.join(DATA_DIR, f"{settings['INIT_MONTH']}.{MONTH}")

utils.check_folder(MONTH_DIR)

cluster = SLURMCluster()
cluster.scale(jobs=8)
print(cluster, flush=True)
client = Client(cluster)
print(client, flush=True)

predictors = pd.read_excel(
    os.path.join(settings["MODEL_SRC"], settings["PREDICTORS"]), index_col=[0]
)
pisco = (
    xr.open_dataset(settings["PISCO_DATA"], decode_times=False)
    .rename({"X": "lon", "Y": "lat", "T": "time"})
    .load()
)
pisco.time.attrs["calendar"] = "360_day"
pisco = xr.decode_cf(pisco).Prec
pisco = pisco.sel(time=slice("1981-10-01", "2016-10-01"))

sel_db = predictors.loc[1981:2015].copy()
months_index = pisco.groupby("time.month").groups


stepwise_selection = delayed(base.stepwise_selection)

full_model = {}

sel_db_model = sel_db.reset_index(drop=True)

for mnum, mindex in months_index.items():
    print(mnum, flush=True)
    full_model[mnum] = [
        (
            (lat, lon),
            stepwise_selection(
                sel_db_model,
                pisco.isel(time=mindex)
                .sel(lat=lat, lon=lon)
                .to_dataframe()
                .reset_index(drop=True)["Prec"],
                threshold_in=0.05,
                threshold_out=0.1,
                max_vars=12,
                min_vars=4,
                verbose=False,
            ),
        )
        for lat in pisco.lat.data
        for lon in pisco.lon.data
    ]

    print(f"Month number {mnum} ready for computation\n", flush=True)


for mnum, mmodel in full_model.items():
    print(f"\nStarting computation of month number: {mnum}", flush=True)
    res = compute(mmodel)
    print(f"Done computing month number: {mnum}", flush=True)
    print("Starting save", flush=True)
    with open(
        os.path.join(MONTH_DIR, f"model_{MONTH.lower()}.{mnum:02d}.pickle"),
        "wb",
    ) as handle:
        pickle.dump(res[0], handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saving done for month number: {mnum}\n", flush=True)

# Scale down and close cluster
client.close()
cluster.scale(jobs=0)
cluster.close()
