from weighted_ccf_refactored_mod import run_weighted_ccf
import pandas as pd
import numpy as np
from utils import Task, FancyIter, cstr, Message, ccf_mask, FancyArg, Velocity

#### Cross-correlate a single observation ####

spectrum = pd.read_csv('/home/alst3712/exocomet_project/clean_data/2022-04-05_M4368/order_regions/tellurics_corrected_version_0.8.3_2022-04-05_M4368_base_preprocessed_merged.csv')

print(spectrum.head())

model = pd.read_csv('co_template_weighted.csv')

print(model.head())

ccf = run_weighted_ccf(
    spectra=spectrum,
    model=model,
    planet=None,
    species = 'CO',
    vmr = 1e-4,
    exotic_mol= False,
    model_path = model,
    rv_min = 0,
    velocities = np.linspace(-200, 200, 600),
    anti_snr = False
)

