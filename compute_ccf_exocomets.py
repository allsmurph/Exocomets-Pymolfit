#%%
from weighted_ccf_refactored_mod import run_weighted_ccf
import pandas as pd
import numpy as np
from utils import Task, FancyIter, cstr, Message, ccf_mask, FancyArg, Velocity
from cross_correlation_mod import cross_correlate
import matplotlib.pyplot as plt
import glob
import re


###################################
######## Define weights ###########
###################################


##############################################
#### Cross-correlate a single observation ####
##############################################


## load spectrum

#spectrum = pd.read_csv('/home/alst3712/exocomet_project/clean_data/2022-04-05_M4368/order_regions/tellurics_corrected_version_0.8.3_2022-04-05_M4368_base_preprocessed_merged.csv')
spectrum = pd.read_csv('/home/alst3712/exocomet_project/clean_data/2022-04-05_M4368/order_regions/tellurics_corrected_version_0.8.3_2022-04-05_M4368_base_preprocessed.csv')

print(spectrum.head())

spectrum_sorted = (
    spectrum
    .replace([np.inf, -np.inf], np.nan)
    .dropna(subset=['wavelength', 'flux.deviation'])
    .sort_values('wavelength')
    .drop_duplicates(subset='wavelength', keep='first')
    .reset_index(drop=True)
)

plt.plot(spectrum_sorted['wavelength'], spectrum_sorted['flux.deviation'])
plt.show()

#%% load model

model = pd.read_csv('/home/alst3712/exocomet_project/Opacity_spectra/CO_absorptions/CO_model_1000K_1e+18_0.9cf_weighted.csv')
# model_files = glob.glob('Opacity_spectra/CO_absorptions/CO_model_*.csv')

# models = []

# for file in model_files:
model = (
    model
    .replace([np.inf, -np.inf], np.nan)
    .dropna(subset=['wavelength', 'flux'])
    .sort_values('wavelength')
    .drop_duplicates(subset='wavelength', keep='first'))
#models.append(model)

plt.plot(model['wavelength'][10:-2], model['flux'][10:-2])
#print(f'Will cross-correlate with {len(models)} models')

#%%

print(np.all(np.diff(spectrum_sorted['wavelength']) > 0))
print(np.all(np.diff(model['wavelength']) > 0))
#%%
velocities = np.linspace(-300, 300, 1000)


# for m, f in zip(models[:], model_files[:]):
#     match = re.search(r'(\d+K_.*)$', f)
#     extracted = match.group(1)

#     print(extracted)    
#     ccf = cross_correlate(m, m, velocities,
#                             x_axis=['wavelength', 'wavelength'],
#                             y_axis=['flux', 'f'])
#     plt.plot(velocities, ccf)
#     plt.title(f'Using model {extracted}')
#     plt.show()

error = spectrum['error']
model_interp = np.interp(spectrum[f'wavelength'], model['wavelength'], model['weight.smooth'])
weight = model_interp / error*2

ccf = cross_correlate(spectrum_sorted, model, velocities,# weight=weight,
                        x_axis=['wavelength', 'wavelength'],
                        y_axis=['flux.no_baseline', 'flux.no_baseline'])
plt.plot(velocities, ccf)
#plt.title(f'Using model {extracted}')
plt.show()
