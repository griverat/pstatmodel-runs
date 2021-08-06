import argparse
import os

import pstatmodel as psm
from dmelon import utils

parser = argparse.ArgumentParser(
    description="Create database with the default variables"
)
parser.add_argument("settings", type=str)
args = parser.parse_args()

settings = args.settings
settings = utils.load_json(settings)


ModelPredictor = psm.ModelVariables()
ModelPredictor.shiftAllVariables(
    init_month=settings["INIT_MONTH"], fyear=settings["FYEAR"]
)
model_init_data = ModelPredictor.get_datatable()
model_init_data.dropna(axis=1, how="all").to_excel(
    os.path.join(settings["MODEL_SRC"], settings["PREDICTORS"])
)
