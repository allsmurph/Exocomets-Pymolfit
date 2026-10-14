from pathlib import Path

import matplotlib.pyplot as plt
from pymolfit import correct

### To correct one of the many orders you do the following.


# grab the file from one location...
FILE = Path(
    "/srv/local/anba6163/Clean_Data/2022-04-05_M4368/extr/cr2res_obs_nodding_extracted_combined.fits"
)

date_dir = FILE.parent.parent.name

#...but save the order regions elsewhere
REGION_DIR = Path("/home/alst3712/clean_data") / date_dir / "order_regions"

REGION_DIR.mkdir(parents=True, exist_ok=True)

print(REGION_DIR)

# result = correct(
#     input_path=file,
#     hdu=1,
#     wavelength_col="03_01_WL",
#     flux_col="03_01_SPEC",
#     uncertainty_col="03_01_ERR",
#     wavelength_unit="nm",
#     wavelength_medium="vacuum",
# )

# observed = result.spectrum.to_unit("nm")
# corrected = result.corrected.to_unit("nm")
# fig, ax = plt.subplots(figsize=(12, 5))
# ax.plot(observed.wavelength, observed.flux, color="0.5", label="Uncorrected")
# ax.plot(corrected.wavelength, corrected.flux, label="Corrected")
# ax.set(xlabel="Vacuum wavelength [nm]", ylabel="Flux", title="Detector 1, order 03_01")
# ax.legend()
# fig.tight_layout()
# plt.show()

### to get a better fit you have to define fit regions, for which the fitting parameters are approximated from

from pymolfit import load_spectrum, select_telluric_regions



# this is where the region is going to be defined and saved
region_file = REGION_DIR / "fitting_regions_test.ecsv"

# note: need to specify wavelength and flux columns
spectrum = load_spectrum(
    FILE,
    hdu=1,
    wavelength_col="03_01_WL",
    flux_col="03_01_SPEC",
    uncertainty_col="03_01_ERR",
    wavelength_unit="nm",
    wavelength_medium="vacuum",
    save_header=False,
)

        
    
select_telluric_regions(spectrum, output_path=region_file)

result = correct(
    input_path=FILE,
    hdu=1,
    wavelength_col="03_01_WL",
    flux_col="03_01_SPEC",
    uncertainty_col="03_01_ERR",
    wavelength_unit="nm",
    wavelength_medium="vacuum",
    region_file=region_file,
)

observed = result.spectrum.to_unit("nm")
corrected = result.corrected.to_unit("nm")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(observed.wavelength, observed.flux, color="0.5", label="Uncorrected")
ax.plot(corrected.wavelength, corrected.flux, label="Corrected")
ax.set(xlabel="Vacuum wavelength [nm]", ylabel="Flux", title="Detector 1, order 03_01")
ax.legend()
fig.tight_layout()
plt.show()
plt.savefig(REGION_DIR / "order_03_01_correction.png", dpi=300)

### Check the other script to see how to correct all orders at once. I also made parameter changes so that you will get a better correction.

