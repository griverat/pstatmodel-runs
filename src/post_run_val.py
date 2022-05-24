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
DATA_DIR = settings["DATA_DIR"]
MONTH_DIR = os.path.join(DATA_DIR, f"{settings['INIT_MONTH']}.{MONTH}")
VALIDATION_DIR = os.path.join(MONTH_DIR, "validation")
NC_DIR = os.path.join(MONTH_DIR, "Data")

utils.check_folder(VALIDATION_DIR)
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

lats = pisco.lat.data
lons = pisco.lon.data

pred_data_val = xr.DataArray(
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

new_pred = predictors.loc[1981:2015].copy()
new_pred["const"] = 1


for val_year in range(1982, 2017):
    try:
        with open(
            os.path.join(VALIDATION_DIR, f"full_model_val.{val_year}.pickle"),
            "rb",
        ) as handle:
            full_model_val = pickle.load(handle)
        print(f"Succesfully read validation model for year {val_year}", flush=True)

        print(f"\nStarting computation for year {val_year}", flush=True)
        for mnum, nmodel in full_model_val.items():
            if mnum in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
                sel_time = f"{val_year}-{mnum}-15"
            elif mnum in [10, 11, 12]:
                sel_time = f"{val_year-1}-{mnum}-15"
            sel_time = pred_data_val.sel(time=sel_time).time.data
            for (lat, lon), (pixel_vars, pixel_model) in nmodel:
                pred_data_val.loc[
                    dict(lat=lat, lon=lon, time=sel_time)
                ] = pixel_model.predict(
                    new_pred.loc[val_year - 1, pixel_model.params.index].values
                )[
                    0
                ]
            print(f"Done month number {mnum}", flush=True)
        print(f"Done computation for validation year {val_year}", flush=True)

    except FileNotFoundError:
        print(f"Couldn't find model for val year {val_year}", flush=True)


pred_data_val = pred_data_val.dropna(dim="time", how="all")

pred_data_val.name = "pred_data_val"
pred_data_val.to_netcdf(os.path.join(NC_DIR, "pred_data_val.nc"))
