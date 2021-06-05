#%%
import pandas as pd
import pstatmodel as psm
import pstatmodel.utils as psu

#%%
ModelPredictor = psm.ModelVariables()
#%%
ecData = pd.read_csv("ec_ersstv5.txt", parse_dates=[0])
ModelPredictor.register_variable("EC_index", ["E", "C"], ecData)

#%%
shift_kwargs = dict(init_month="09", fyear=2022)
ModelPredictor.shiftAllVariables(**shift_kwargs)

model_init_data = ModelPredictor.get_datatable()

# %%
model_init_data.dropna(axis=1, how="all").to_excel("Predictores_IniSep.xlsx")
# %%
