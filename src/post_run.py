#%%
import argparse
import os
import pickle

import numpy as np
import pandas as pd
import xarray as xr
from dmelon import utils

#%%
parser = argparse.ArgumentParser(description="Compute model output")
parser.add_argument("settings", type=str)
args = parser.parse_args()

settings = args.settings
settings = utils.load_json(settings)

#%%
MONTH = settings["MONTH"]
DATA_DIR = settings["DATA_DIR"]
MONTH_DIR = os.path.join(DATA_DIR, f"{settings['INIT_MONTH']}.{MONTH}")
VALIDATION_DIR = os.path.join(MONTH_DIR, "validation")
NC_DIR = os.path.join(MONTH_DIR, "Data")

utils.check_folder(VALIDATION_DIR)
utils.check_folder(NC_DIR)


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
pisco = pisco.sel(time=slice("1981-10-01", "2016-10-01"))

months_index = pisco.groupby("time.month").groups

#%%
full_model = {}
for mnum, mindex in months_index.items():
    try:
        with open(
            os.path.join(MONTH_DIR, f"model_{MONTH.lower()}.{mnum:02d}.pickle"),
            "rb",
        ) as handle:
            full_model[mnum] = pickle.load(handle)
        print(f"Succesfully read model for month number {mnum}", flush=True)
    except:
        print(f"Couldn't find model for month number {mnum}", flush=True)

#%%

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

#%%

pred_groups = pred_data.groupby("time.month").groups
new_pred = predictors.loc[1981:2015].copy()
new_pred["const"] = 1

#%%

for mnum, nmodel in full_model.items():

    print(f"\nStarting model month number: {mnum}", flush=True)

    for (lat, lon), (pixel_vars, pixel_model, thresh_in) in nmodel:
        if not isinstance(pixel_model, float) and len(pixel_vars) != 0:
            #             pixel_model, thresh_in = pixel_model
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

# del full_model
#%%
pred_data = pred_data.dropna(dim="time", how="all")
metric_data = metric_data.dropna(dim="month", how="all")
metric2_data = metric2_data.dropna(dim="month", how="all")
nvar_data = nvar_data.dropna(dim="month", how="all")

#%%
full_model_val = {}
for val_year in range(1982, 2017):
    try:
        with open(
            os.path.join(VALIDATION_DIR, f"full_model_val.{val_year}.pickle"),
            "rb",
        ) as handle:
            full_model_val[val_year] = pickle.load(handle)
        print(f"Succesfully read validation model for year {val_year}", flush=True)
    except:
        print(f"Couldn't find model for val year {val_year}", flush=True)

#%%
for year, validation_model in full_model_val.items():
    print(f"\nStarting computation for year {year}", flush=True)
    for mnum, nmodel in validation_model.items():
        if mnum in [1, 2, 3, 4]:
            sel_time = f"{year}-{mnum}-15"
        elif mnum in [10, 11, 12]:
            sel_time = f"{year-1}-{mnum}-15"
        sel_time = pred_data_val.sel(time=sel_time).time.data
        for (lat, lon), (pixel_vars, pixel_model) in nmodel:
            pred_data_val.loc[
                dict(lat=lat, lon=lon, time=sel_time)
            ] = pixel_model.predict(
                new_pred.loc[year - 1, pixel_model.params.index].values
            )[
                0
            ]
        print(f"Done month number {mnum}", flush=True)
    print(f"Done computation for validation year {year}", flush=True)
    #             metric_data_val.loc[
    #                 dict(val_year=year, lat=lat, lon=lon, month=mnum)
    #             ] = pixel_model.rsquared
    #             metric2_data_val.loc[
    #                 dict(val_year=year, lat=lat, lon=lon, month=mnum)
    #             ] = pixel_model.rsquared_adj

#%%

pred_data_val = pred_data_val.dropna(dim="time", how="all")

#%%
pred_data.name = "pred_data"
pred_data.to_netcdf(os.path.join(NC_DIR, "pred_data.nc"))

metric_data.name = "metric_data"
metric_data.to_netcdf(os.path.join(NC_DIR, "metric_data.nc"))

metric2_data.name = "metric2_data"
metric2_data.to_netcdf(os.path.join(NC_DIR, "metric2_data.nc"))

nvar_data.name = "nvar_data"
nvar_data.to_netcdf(os.path.join(NC_DIR, "nvar_data.nc"))

pred_data_val.name = "pred_data_val"
pred_data_val.to_netcdf(os.path.join(NC_DIR, "pred_data_val.nc"))

thresh_data.name = "thresh_data"
thresh_data.to_netcdf(os.path.join(NC_DIR, "thresh_data.nc"))

model_data.name = "model_data"
with open(os.path.join(NC_DIR, "model_data.pickle"), "wb") as handle:
    pickle.dump(model_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
