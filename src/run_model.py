import argparse
import logging
import os
import pickle
import sys

import pandas as pd
from dask import compute, delayed
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from dmelon import utils
from pstatmodel.stepwise import base

import xarray as xr

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Run the pstatmodel")
parser.add_argument("settings", type=str)
args = parser.parse_args()

settings = args.settings
settings = utils.load_json(settings)

MONTH = settings["MONTH"]
MONTH_LIST = settings["MLIST"]
DATA_DIR = settings["DATA_DIR"]
MONTH_DIR = os.path.join(DATA_DIR, f"{settings['INIT_MONTH']}.{MONTH}")

utils.check_folder(MONTH_DIR)

cluster = SLURMCluster()
cluster.scale(jobs=8)
logger.info(cluster)
client = Client(cluster)
logger.info(client)

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
months_index = {k: v for k, v in months_index.items() if k in MONTH_LIST}

stepwise_selection = delayed(base.stepwise_selection)

full_model = {}

sel_db_model = delayed(sel_db.reset_index(drop=True))

for mnum, mindex in months_index.items():
    logger.info(f"Building month {mnum} futures")

    _pisco_sel = pisco.isel(time=mindex)
    full_model[mnum] = []
    for lat in pisco.lat.data:
        _pisco_sel_lat = _pisco_sel.sel(lat=lat)
        for lon in pisco.lon.data:
            _pisco_sel_lat_lon = delayed(
                _pisco_sel_lat.sel(lon=lon)
                .to_dataframe()
                .reset_index(drop=True)["Prec"]
            )
            full_model[mnum].append(
                stepwise_selection(
                    sel_db_model,
                    _pisco_sel_lat_lon,
                    threshold_in=0.05,
                    threshold_out=0.1,
                    max_vars=12,
                    min_vars=4,
                    verbose=False,
                )
            )

    logger.info(f"Month number {mnum} ready for computation\n")


for mnum, mmodel in full_model.items():
    logger.info(f"Starting computation of month number: {mnum}")
    res = compute(mmodel)
    logger.info(f"Done computing month number: {mnum}")

    logger.info("Starting save")
    out_file = os.path.join(MONTH_DIR, f"model_{MONTH.lower()}.{mnum:02d}.pickle")
    logger.info(f"Output file: {out_file}")

    with open(out_file, "wb") as handle:
        pickle.dump(res[0], handle, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Saving done for month number: {mnum}\n")

# Scale down and close cluster
client.close()
cluster.scale(jobs=0)
cluster.close()
