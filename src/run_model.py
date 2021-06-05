#%%
import argparse
import os
import pickle

import pandas as pd
import statsmodels.api as sm
import xarray as xr
from dask import compute, delayed
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from dmelon import utils
from pstatmodel.stepwise import base

#%%
parser = argparse.ArgumentParser(description="Run the pstatmodel")
parser.add_argument("settings", type=str)
args = parser.parse_args()

settings = args["settings"]
settings = utils.load_json(settings)

#%%
MONTH = settings["MONTH"]
DATA_DIR = settings["DATA_DIR"]
MONTH_DIR = os.path.join(DATA_DIR, MONTH)
VALIDATION_DIR = os.path.join(MONTH_DIR, "validation")

utils.check_folder(MONTH)
utils.check_folder(DATA_DIR)
utils.check_folder(MONTH_DIR)
utils.check_folder(VALIDATION_DIR)

#%%
cluster = SLURMCluster()
cluster.scale(jobs=8)
print(cluster, flush=True)
client = Client(cluster)
print(client, flush=True)

#%%
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
pisco = pisco.sel(time=slice("1981-10-01", "2016-05-01"))

#%%
sel_db = predictors.loc[1981:2015].copy()
months_index = pisco.groupby("time.month").groups

#%%

stepwise_selection = delayed(base.stepwise_selection)

full_model = {}

sel_db_model = sel_db.reset_index(drop=True)

for mnum, mindex in months_index.items():
    if mnum in [5, 6, 7, 8, 9]:
        continue
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

#%%

for mnum, mmodel in full_model.items():
    print(f"\nStarting computation of month number: {mnum}", flush=True)
    res = compute(mmodel)
    print(f"Done computing month number: {mnum}", flush=True)
    print("Starting save", flush=True)
    with open(
        os.path.join(DATA_DIR, f"model_{settings['MONTH'].lower()}.{mnum:02d}.pickle"),
        "wb",
    ) as handle:
        pickle.dump(res[0], handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saving done for month number: {mnum}\n", flush=True)

#%%
# Scale down and close cluster
client.close()
cluster.scale(jobs=0)
cluster.close()

#%%
full_model = {}
for mnum, mindex in months_index.items():
    try:
        with open(
            os.path.join(
                DATA_DIR, f"model_{settings['MONTH'].lower()}.{mnum:02d}.pickle"
            ),
            "rb",
        ) as handle:
            full_model[mnum] = pickle.load(handle)
        print(f"Succesfully read model for month number {mnum}", flush=True)
    except:
        print(f"Couldn't find model for month number {mnum}", flush=True)

#%%

# Validation model container
full_model_val = {}
sel_db.loc[:, "const"] = 1

for val_year in range(1982, 2017):

    print(f"\nComputing validation model for year {val_year}", flush=True)

    full_model_val[val_year] = {}
    exclude_time_start = f"{val_year-1}-05-01"
    exclude_time_end = f"{val_year}-10-01"

    # Removing val_year from pisco data

    pisco_val = xr.concat(
        [
            pisco.sel(time=slice(None, exclude_time_start)),
            pisco.sel(time=slice(exclude_time_end, None)),
        ],
        dim="time",
    )

    months_val_index = pisco_val.groupby("time.month").groups

    # Removing val_year - 1 from predictors data
    sel_db_val = sel_db.query("year!=@val_year-1").reset_index(drop=True)

    for mnum, mmodel in full_model.items():
        print(f"Computing month number {mnum}", flush=True)
        result_val = []
        for (lat, lon), (pixel_vars, pixel_model, _) in mmodel:
            if not isinstance(pixel_model, float) and len(pixel_vars) != 0:
                new_model = sm.OLS(
                    pisco_val.isel(time=months_val_index[mnum])
                    .sel(lat=lat, lon=lon)
                    .to_dataframe()
                    .reset_index(drop=True)["Prec"],
                    sel_db_val[pixel_vars + ["const"]],
                ).fit()
                result_val.append(((lat, lon), (pixel_vars, new_model)))
        full_model_val[val_year][mnum] = result_val
    print(f"Done validation year {val_year}\n", flush=True)

#%%
for year_val_save, model_val_save in full_model_val.items():
    print(f"\nStarting save of validation year: {year_val_save}", flush=True)
    with open(
        os.path.join(VALIDATION_DIR, f"full_model_val.{year_val_save}.pickle"),
        "wb",
    ) as handle:
        pickle.dump(model_val_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saving done for validation year: {year_val_save}", flush=True)
