import argparse
import os
import pickle

import numpy as np
import pandas as pd
from dmelon import utils

import xarray as xr

parser = argparse.ArgumentParser(description="Compute model output")
parser.add_argument("settings", type=str)
args = parser.parse_args()

settings = args.settings
settings = utils.load_json(settings)

MONTH = settings["MONTH"]
FYEAR = settings["FYEAR"]
MONTH_LIST = settings["MLIST"]
DATA_DIR = settings["DATA_DIR"]
MONTH_DIR = os.path.join(DATA_DIR, f"{settings['INIT_MONTH']}.{MONTH}")
NC_DIR = os.path.join(DATA_DIR, str(FYEAR), f"{settings['INIT_MONTH']}.{MONTH}", "Data")

utils.check_folder(NC_DIR)
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

months_index = pisco.groupby("time.month").groups
months_index = {k: v for k, v in months_index.items() if k in MONTH_LIST}

full_model = {}
for mnum, mindex in months_index.items():
    try:
        with open(
            os.path.join(MONTH_DIR, f"model_{MONTH.lower()}.{mnum:02d}.pickle"),
            "rb",
        ) as handle:
            full_model[mnum] = pickle.load(handle)
        print(f"Succesfully read model for month number {mnum}", flush=True)
    except FileNotFoundError:
        print(f"Couldn't find model for month number {mnum}", flush=True)

lats = pisco.lat.data
lons = pisco.lon.data

fcst_data = xr.DataArray(
    np.nan,
    coords=[
        (
            "time",
            pd.date_range("1981-10", f"{FYEAR}-09", freq="MS") + pd.DateOffset(days=14),
        ),
        ("lat", lats),
        ("lon", lons),
    ],
)

pred_groups = fcst_data.groupby("time.month").groups
new_pred = predictors.loc[1981:].copy()
new_pred["const"] = 1

for mnum, nmodel in full_model.items():

    print(f"\nStarting model month number: {mnum}", flush=True)

    for (lat, lon), (pixel_vars, pixel_model, thresh_in) in nmodel:
        if not isinstance(pixel_model, float) and len(pixel_vars) != 0:
            sel_time = fcst_data.time.isel(time=pred_groups[mnum]).data
            fcst_data.loc[dict(lat=lat, lon=lon, time=sel_time)] = pixel_model.predict(
                new_pred[pixel_model.params.index]
            )
    print(f"Finished model month number: {mnum}\n")

fcst_data = fcst_data.dropna(dim="time", how="all")

fcst_data.name = "fcst_data"
fcst_data.to_netcdf(os.path.join(NC_DIR, "fcst_data.nc"))
