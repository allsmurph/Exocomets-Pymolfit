#%%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from pymolfit import Spectrum, correct, select_telluric_regions
from plotting_params import use_my_style
import time
use_my_style()
from joblib import Parallel, delayed


start_time = time.time()


# grab the file from one location...
#FILE = Path(
#    "/srv/local/anba6163/Clean_Data/2022-04-05_M4368/extr/cr2res_obs_nodding_extracted_combined.fits"
#)

base_dir = Path("/srv/local/anba6163/Clean_Data/")

fits_files = sorted(
    file for file in base_dir.rglob("cr2res_obs_nodding_extracted_combined.fits")
    if "old" not in file.parts)

observations = []
for file in fits_files:
    date_dir = file.parent.parent.name

    if "M" not in date_dir:
        print(f"Skipping {date_dir}: not M-band")
        continue

    region_dir = (
        Path("/home/alst3712/clean_data")
        / date_dir
        / "order_regions"
    )
    region_dir.mkdir(parents=True, exist_ok=True)

    observations.append((file, region_dir))

print(f"Found {len(observations)} M-band observations")

# %%

#for 1 file, you can do the following to correct all orders in one go:
def load_all_orders(path):
    orders = {}
    with fits.open(path) as hdus:
        for chip, hdu in enumerate(hdus[1:], 1): #for each of the three chips...
            for name in hdu.columns.names:
                if name.endswith("_WL"):
                    #for each of the 5/6 orders...
                    order = name.removesuffix("_WL")
                    orders[f"chip_{chip}_{order}"] = Spectrum(
                        wavelength=hdu.data[name][40:-40], # remove edges, values from trial and error
                        flux=hdu.data[f"{order}_SPEC"][40:-40],
                        uncertainty=hdu.data[f"{order}_ERR"][40:-40],
                        #group_id=np.zeros(hdu.data[name].size, dtype=int),
                        wavelength_unit="nm",
                        wavelength_medium="vacuum",
                        meta={"source": str(path)},
                    )

    return orders


def make_region_file(spectrum, path):
    selector = select_telluric_regions(
        spectrum, show=False,
        automatic_region_half_width_pixels=20, #default is 12
    )

    try:
        selector.add_automatic_fit_regions(400)  # Per order; overlapping windows are merged.
        selector.save(path)
    finally:
        selector.close()

# %%

def process_file(FILE, DIR):
    date_dir = FILE.parent.parent.name
    print(
        f"[PID {os.getpid()}] Starting {date_dir}",
        flush=True,
    )

    orders = load_all_orders(FILE)
    corrected_orders = []

    DIR.mkdir(exist_ok=True)

    output_file = DIR / f"tellurics_corrected_parallelised_{FILE.parent.parent.name}.dat"

    temp_file = output_file.with_suffix(".dat.tmp")
    # if output_file.exists() and os.access(output_file, os.R_OK):
    #     print(f"Corrected spectrum already exists at {output_file}. Skipping correction.")
    #     continue

    for index, (name, spectrum) in enumerate(orders.items(), 1):

        print(
                f"[PID {os.getpid()}] "
                f"{date_dir}: order {index}/{len(orders)} "
                f"({name})",
                flush=True,
            )
        
        region_file = DIR / f"{name}.ecsv" #note to self - ecsv can store stuff like units and can be used to write astropy tables
        make_region_file(spectrum, region_file)

        wave = np.asarray(spectrum.wavelength)
        fluxvals = np.asarray(spectrum.flux)

        print(f"flux range = {np.nanmin(fluxvals):.3f}–{np.nanmax(fluxvals):.3f} ")
        print(f'last few values are {fluxvals[len(fluxvals)-15:]}')
        print(
            f"{name}: "
            f"spectrum = {np.nanmin(wave):.3f}–{np.nanmax(wave):.3f} "
            f"{spectrum.wavelength_unit}, "
            f"regions = {region_file}"
        )

        try:
            result = correct(
                spectrum=spectrum, 
                region_file=region_file, 
                auto_segment=True, #``True`` separates FITS orders/detectors and wavelength discontinuities into physical groups, then divides wide groups into bounded radiative-transfer chunks. Molecular columns remain global, continuum coefficients remain local to numerical chunks, and automatic wavelength polynomials are shared across every chunk from the same physical group. ``False`` fits the input as one segment and may exceed grid limits.
                segment_size=0.01, # in microns, maximum numerical radiative-transfer chunk width in microns
                continuum_order=2, #
                solve_continuum_linear=True, #`True`` always selects this stable profiled approach, while ``False`` includes continuum coefficients in the nonlinear parameter vector
                loss="soft_l1", # soft_l1 is standard in new version; Controls how residuals influence the fit. ``linear`` is ordinary squared-residual least squares and is the statistically preferred default for a clean, well-masked spectrum with reliable uncertainties. ``soft_l1`` is usually appropriate for a mostly clean spectrum containing a limited number of cosmic rays, bad pixels, or unmasked spectral features because it reduces their influence without ignoring normal residuals. ``huber`` is another moderate robust loss, while ``cauchy`` and ``arctan`` suppress large outliers more strongly. Robust loss cannot repair generally poor calibration, wavelength misalignment, incorrect uncertainties, or an unsuitable atmospheric model.
                f_scale=1.0, #Residual scale separating normal points from downweighted outliers for robust losses;
                fit_wavelength_shift=False, 
                fit_segment_wavelength_polynomial=True, # ``True`` fits one wavelength-correction polynomial per detected physical order/detector group and shares it across that group's numerical chunks; use this when line residuals show within-order wavelength distortion, and do not combine it with global or constant per-group wavelength fitting.
                segment_wavelength_polynomial_order=2, #Degree of each physical-group wavelength polynomial
                wavelength_shift_bounds=(-0.0002, 0.0002),  # Microns: +/-0.2 nm. ``None`` uses ``(-3, 3)`` detector pixels for automatic model selection or the legacy micron bounds for explicit wavelength models; supplied values are pixels in automatic mode and microns in explicit mode.
                lsf_sigma_pixels="auto", #
                fit_lsf_sigma=True, # ``"auto"`` refines an automatically estimated ``lsf_sigma_pixels`` when telluric lines are available but keeps an explicitly numeric sigma fixed; ``True`` always fits Gaussian sigma and ``False`` always keeps the resolved value fixed.
                high_resolution_grid=True, # ``True`` computes, convolves, and rebins an oversampled internal model; ``False`` evaluates directly at observed samples.
                high_resolution_oversampling=5, #Approximate internal samples per observed pixel when ``high_resolution_grid=True``.
                min_transmission=0.1,  # Leave near-opaque absorption uncorrected (NaN).
            )

        except Exception as exc:
                print(
                    f"[PID {os.getpid()}] FAILED: "
                    f"file={date_dir}, order={name}, "
                    f"error={type(exc).__name__}: {exc}",
                    flush=True,
                )
                continue

        corrected = result.corrected.to_unit("nm")

        corrected_orders.append(np.column_stack((
                                corrected.wavelength,
                                corrected.flux,
                                corrected.uncertainty)))

    corrected_data = np.vstack(corrected_orders)

    with open(temp_file, "w") as f:
        f.write("# wavelength flux error\n")

        for wavelength, flux, error in corrected_data:
                    f.write(f"{wavelength:.10f} {flux:.10e} {error:.10e}\n")

    os.replace(temp_file, output_file)

    print(f"Saved corrected spectrum to {output_file}")

    return f'Success! {date_dir} completed'


n_jobs = 2
if __name__ == '__main__':
    if observations:
        results = Parallel(n_jobs = n_jobs)(delayed(process_file)(FILE, DIR) for FILE, DIR in observations)

    print(
        f"\nTotal time taken: "
        f"{(time.time() - start_time) / 60:.2f} minutes"
    )
