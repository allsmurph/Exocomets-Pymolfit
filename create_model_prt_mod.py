

###############
### Imports ###
############### 

import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from astropy import units as u
while not os.path.exists('utils'):
    os.chdir('..')
# print('Current working directory:', os.getcwd())
sys.path.append(os.getcwd())

from utils import *
from pathlib import Path

INFO_TRESHOLD = 0.1
BASELINE_KERNEL = 40 # nm # for IPSA # 20 is good to follow a bit the oscillations, 40 may be smoother


# ############## !!!!!!!!!!! ##############
'''Message('Cose illegali')'''

#flux must be in cgs I think, but maybe it doesn't matter?
import glob

# model_files = glob.glob('Opacity_spectra/CO_absorptions/CO_model_*.csv')

# models = []

# for file in model_files:
#     model = pd.read_csv(file)
#     model = (
#         model
#         .replace([np.inf, -np.inf], np.nan)
#         .dropna(subset=['wavelength', 'flux'])
#         .sort_values('wavelength')
#         .drop_duplicates(subset='wavelength', keep='first'))
#     models.append(model)

# for model_data in models:
#     wavelengths = model_data['wavelength']
#     flux = model_data['flux']


file = Path('/home/alst3712/exocomet_project/Opacity_spectra/CO_absorptions/CO_model_1000K_1e+18_0.9cf.csv')
model = pd.read_csv(file)
output_name = file.with_name(file.stem + '_weighted.csv')


spectrum_model = pd.DataFrame({
    'wavelength': model['wavelength'],
    'flux': model['flux'] #normalised and broadened already
})

# ########################
# ### Baseline removal ###
# ########################

with Task(f"Removing the baseline with {cstr('IPSA'):b}...", new_line = False):
    from pybaselines import Baseline

    baseline, _ = Baseline().ipsa(
        spectrum_model['flux'],
        half_window = int(BASELINE_KERNEL / spectrum_model['wavelength'].diff().median()) # 10 nm kernel window
    )

    spectrum_model["flux.no_baseline"] = spectrum_model["flux"] - baseline

##################################
### Compute information weight ###
##################################

# this has to be done before contrast computation - why???? I changed it

# compute weight as derivative of abs value

step = spectrum_model["wavelength"].diff().median()
#contrast_no_baseline = spectrum['contrast']-baseline/stellar_spectrum['flux']
spectrum_model['weight'] = spectrum_model['flux'].diff().abs().fillna(0) # first weight is NaN because of diff

# smooth out the result
spectrum_model['weight.smooth'] = spectrum_model['weight'].rolling(
    int(5 / step), # 5nm window
    center=True
).sum().fillna(0)

spectrum_model['weight.smooth'] = spectrum_model['weight.smooth'].rolling(
    int(5 / step),
    center=True
).median().fillna(0)

# rescale for plot and mask
spectrum_model['weight'] /= spectrum_model['weight'].std()
spectrum_model['weight.smooth'] /= spectrum_model['weight.smooth'].std()
spectrum_model['weight.mask'] = (spectrum_model['weight.smooth'] > INFO_TRESHOLD).astype(int)

spectrum_model.to_csv(output_name, index=False)

#######################
### Save the result ###
#######################

# filepath = os.path.join(
#         FOLDER,
#         OUTPUT_FILENAME
#     )

# with Task(f"Saving the model to: {cstr(filepath):b}", new_line=False):
#     spectrum.to_csv(filepath, index=False)
# FancyArg.set_return(filepath)

# Message.par()
# Message("Summary of the content of the model (columns of the dataframe):")

# summary = f"""{cstr("'wavelength'"):c} [nm]
# {cstr("'flux'"):c} [cgs/cm]
# {cstr("'contrast.no_baseline'"):c} [?]: baseline of the spectrum is computed and removed. {cstr("'flux.baseline'= 'flux'-'flux.no_baseline'"):y} can be computed from here.
# {cstr("'contrast'"):c} [?]
# {cstr("'weight'"):c} [?]: absolute value of derivative of the contrast.
# {cstr("'weight.smooth'"):c} [?]: smoothed version of the weight  (rolling sum and median over 5nm window each).
# {cstr("'weight.mask'"):c} [bool]: 0 where the signal is flat, 1 where it isn't.
# {cstr("'stellar_flux'"):c} [cgs/cm]: stellar flux at 1AU, scaled to the planet radius. {cstr("'contrast'='flux'/'stellar_flux'"):y} gives contrast at 19.3 pc and {cstr("'contrast.no_baseline'='contrast'-('flux.baseline'/'stellar_flux')"):y} gives the contrast without the stellar baseline.
# {cstr("'stellar_flux.baseline'"):c} [cgs/cm]: baseline of the stellar flux. Might not be usefull
# """

# with Message.tab():
#     for line in summary.split('\n'):
#         if not line:
#             continue
#         Message.print(line)

# Message.par()

# '''suggest_next_command(
#     pythonfile="scripts/weighted_ccf.py",
#     args={
#         "f": f"data/CRIRES/CRIRES_{PLANET}",
#         "model": filepath,
#         "show": True
#     }
# )'''
