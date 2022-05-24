import argparse
import os
import pickle

import numpy as np
import pandas as pd
import xarray as xr
from dask import compute, delayed
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from dmelon import utils
from pstatmodel.stepwise import base

parser = argparse.ArgumentParser(description="Run the pstatmodel")
parser.add_argument("settings", type=str)
args = parser.parse_args()

settings = args.settings
settings = utils.load_json(settings)

MONTH = settings["MONTH"]
DATA_DIR = settings["DATA_DIR"]
MONTH_DIR = os.path.join(DATA_DIR, f"{settings['INIT_MONTH']}.{MONTH}")
NC_TESTS_DIR = os.path.join(MONTH_DIR, "tests")

utils.check_folder(NC_TESTS_DIR)

cluster = SLURMCluster()
cluster.scale(jobs=4)
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
    full_model[mnum] = [
        (
            (lat, lon),
            stepwise_selection(
                sel_db_model.iloc[:-1],
                pisco.isel(time=mindex[:-1])
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
        os.path.join(MONTH_DIR, f"model_{MONTH.lower()}.2015-2016.{mnum:02d}.pickle"),
        "wb",
    ) as handle:
        pickle.dump(res[0], handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saving done for month number: {mnum}\n", flush=True)

# Scale down and close cluster
client.close()
cluster.scale(jobs=0)
cluster.close()

full_model = {}
for mnum, mindex in months_index.items():
    try:
        with open(
            os.path.join(
                MONTH_DIR, f"model_{MONTH.lower()}.2015-2016.{mnum:02d}.pickle"
            ),
            "rb",
        ) as handle:
            full_model[mnum] = pickle.load(handle)
        print(f"Succesfully read model for month number {mnum}", flush=True)
    except:
        print(f"Couldn't find model for month number {mnum}", flush=True)

lats = pisco.lat.data
lons = pisco.lon.data

pred_data = xr.DataArray(
    np.nan,
    coords=[
        (
            "time",
            pd.date_range("1981-10", "2016-09", freq="MS") + pd.DateOffset(days=14),
        ),
        ("lat", lats),
        ("lon", lons),
    ],
)


metric_data = xr.DataArray(
    np.nan,
    coords=[
        (
            "month",
            np.arange(1, 13),
        ),
        ("lat", lats),
        ("lon", lons),
    ],
)

metric2_data = metric_data.copy()

model_data = metric_data.copy().astype(object)

nvar_data = metric_data.copy()
thresh_data = metric_data.copy()

pred_data_val = pred_data.copy()

pred_groups = pred_data.groupby("time.month").groups
new_pred = predictors.loc[1981:2015].copy()
new_pred["const"] = 1

for mnum, nmodel in full_model.items():

    print(f"\nStarting model month number: {mnum}", flush=True)

    for (lat, lon), (pixel_vars, pixel_model, thresh_in) in nmodel:
        if not isinstance(pixel_model, float) and len(pixel_vars) != 0:
            sel_time = pred_data.time.isel(time=pred_groups[mnum]).data
            pred_data.loc[dict(lat=lat, lon=lon, time=sel_time)] = pixel_model.predict(
                new_pred[pixel_model.params.index]
            )
            metric_data.loc[dict(lat=lat, lon=lon, month=mnum)] = pixel_model.rsquared
            metric2_data.loc[
                dict(lat=lat, lon=lon, month=mnum)
            ] = pixel_model.rsquared_adj
            thresh_data.loc[dict(lat=lat, lon=lon, month=mnum)] = np.round(
                thresh_in, decimals=2
            )
            nvar_data.loc[dict(lat=lat, lon=lon, month=mnum)] = len(pixel_vars)
            model_data.loc[dict(lat=lat, lon=lon, month=mnum)] = pixel_model

    print(f"Finished model month number: {mnum}\n")

pred_data = pred_data.dropna(dim="time", how="all")
metric_data = metric_data.dropna(dim="month", how="all")
metric2_data = metric2_data.dropna(dim="month", how="all")
nvar_data = nvar_data.dropna(dim="month", how="all")

pred_data.name = "pred_data"
pred_data.to_netcdf(os.path.join(NC_TESTS_DIR, "pred_data.nc"))

metric_data.name = "metric_data"
metric_data.to_netcdf(os.path.join(NC_TESTS_DIR, "metric_data.nc"))

metric2_data.name = "metric2_data"
metric2_data.to_netcdf(os.path.join(NC_TESTS_DIR, "metric2_data.nc"))

nvar_data.name = "nvar_data"
nvar_data.to_netcdf(os.path.join(NC_TESTS_DIR, "nvar_data.nc"))

pred_data_val.name = "pred_data_val"
pred_data_val.to_netcdf(os.path.join(NC_TESTS_DIR, "pred_data_val.nc"))

thresh_data.name = "thresh_data"
thresh_data.to_netcdf(os.path.join(NC_TESTS_DIR, "thresh_data.nc"))

model_data.name = "model_data"
with open(os.path.join(NC_TESTS_DIR, "model_data.pickle"), "wb") as handle:
    pickle.dump(model_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
