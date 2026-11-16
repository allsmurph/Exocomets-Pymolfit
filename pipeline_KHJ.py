### Andrea Barone 2026, based on Tomas Stolker's pipeline (https://github.com/tomasstolker/pycrires/tree/main) ###

#%%from pathlib import Path
import shutil
import sys, os, warnings, traceback, time, subprocess
from utils import *
from pathlib import Path
from astropy.io import fits
import importlib.metadata as md
package_name = "pycrires"
dist = md.distribution(package_name)
sys.path.insert(1, dist.locate_file(""))
import pycrires
# from whereistheplanet import predict_planet
from astropy.time import Time
from scipy.optimize import curve_fit
from scipy.interpolate import make_interp_spline
import numpy as np
#%%
start_time = time.time()

# FUNCTIONS
def Gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

# LOAD ARGUMENTS
RAW_ROOT = Path('/srv/local/anba6163/Raw_Data')
OUT = '/home/alst3712/exocomet_project/clean_data'
VERBOSE = True
EXTRACTION = False
WORK = '/home/alst3712/exocomet_project/WORK'   # writable scratch folder where each date is copied and reduced

folders = sorted(str(p) for p in RAW_ROOT.iterdir() if p.is_dir() and (p / 'raw').is_dir())

assert len(folders) > 0, f"No date folders with a 'raw' sub-folder found in {RAW_ROOT}"
print(f"Found {len(folders)} date folders to process:")
for f in folders:
    print(f"  {f}")
#%%

def process(folder):
    print(f"Processing folder: {folder}")
    date_name = os.path.basename(os.path.normpath(folder))
    work_dir = os.path.join(WORK, date_name)
    shutil.copytree(folder, work_dir, copy_function=os.symlink, dirs_exist_ok=True)
    subprocess.run(['find', work_dir, '-type', 'd', '-exec', 'chmod', 'u+w', '{}', '+'], check=True)   # copies inherit read-only permissions
    os.chdir(work_dir)
    cwd = os.getcwd()

    pipeline = pycrires.Pipeline('./')

    # DATA CLEANING
    pipeline.rename_files()
    pipeline.extract_header()
    pipeline.cal_dark(verbose=VERBOSE, create_sof=True)
    pipeline.util_calib(calib_type='flat', verbose=VERBOSE, create_sof=True)
    pipeline.util_trace(plot_trace=True, verbose=VERBOSE, create_sof=True)
    pipeline.util_slit_curv(plot_trace=False, verbose=VERBOSE, create_sof=True)
    pipeline.util_extract(calib_type='flat', verbose=VERBOSE, create_sof=True)
    pipeline.util_normflat(verbose=VERBOSE, create_sof=True)
    #pipeline.util_bpm_merge(verbose=VERBOSE, create_sof=True)   # Thomas doesn't use it
    pipeline.util_calib(calib_type='une', verbose=VERBOSE, create_sof=True)
    pipeline.util_extract(calib_type='une', verbose=VERBOSE, create_sof=True)
    pipeline.util_genlines(verbose=VERBOSE, create_sof=True)
    pipeline.util_wave(calib_type='une', poly_deg=0, wl_err=0.1, verbose=VERBOSE, create_sof=True)
    pipeline.util_wave(calib_type='une', poly_deg=2, wl_err=0.03, verbose=VERBOSE, create_sof=True)
    pipeline.util_calib(calib_type='fpet', verbose=VERBOSE, create_sof=True)
    pipeline.util_extract(calib_type='fpet', verbose=VERBOSE, create_sof=True)
    pipeline.util_wave(calib_type='fpet', poly_deg=4, wl_err=0.01, verbose=VERBOSE, create_sof=True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            pipeline.obs_nodding(correct_bad_pixels=True, extraction_required=EXTRACTION, check_existing=False, verbose=VERBOSE, create_sof=True)
    except UserWarning:
        pipeline.obs_nodding_irregular(unique_pairs=False, correct_bad_pixels=True, extraction_required=EXTRACTION, check_existing=False, verbose=VERBOSE, create_sof=True)
    pipeline.run_skycalc(pwv=1.)
    pipeline.plot_spectra(nod_ab='A', telluric=True, corrected=False, file_id=0)

    # SPECTRA EXTRACTION
    pipeline.custom_extract_2d(nod_ab='A', max_separation=2.0)
    pipeline.fit_gaussian(nod_ab='A', extraction_input='custom_extract_2d')
    pipeline.correct_wavelengths_2d(nod_ab='A', input_folder='fit_gaussian', collapse_spatially=False, collapse_exposures=False, create_plots=True, accuracy=0.02)
    pipeline.correct_wavelengths_2d(nod_ab='A', input_folder='correct_wavelengths_2d', collapse_spatially=True, collapse_exposures=True, accuracy=0.005)
    pipeline.correct_wavelengths_2d(nod_ab='A', input_folder='correct_wavelengths_2d', collapse_spatially=True, collapse_exposures=True, accuracy=0.001)

    # STELLAR SPECTRA EXTRACTION
    print("-------------------------\nStellar spectra extraction\n-------------------------")
    hdu = fits.open(cwd+"/product/correct_wavelengths_2d/spectra_nod_A_000_center_corr.fits")
    total_rows = hdu['SPEC'].data.shape[2]
    with Task("Merging all stellar files", new_line=True):
        files = [cwd+"/product/correct_wavelengths_2d/"+f for f in os.listdir(cwd+"/product/correct_wavelengths_2d") if (os.path.isfile(os.path.join(cwd+"/product/correct_wavelengths_2d", f))) and (f.endswith(".fits"))]
        Message(f"Found {cstr(len(files)):c} to process", "i")
        for i, file in enumerate(files):
            hdu = fits.open(file)
            total_rows = hdu['SPEC'].data.shape[2]
            central_rows = list(np.arange(start=int(np.round((total_rows-1)/2))-2, stop=int(np.round((total_rows-1)/2))+3, step=1, dtype=int))
            spec = hdu['SPEC'].data
            err = hdu['ERR'].data
            star_spec = spec[:,:,central_rows,:] # det, order, rows, columns
            star_err = err[:,:,central_rows,:]
            if(i==0):
                wave = hdu['WAVE'].data
                ref_wave = wave[:,:,central_rows,:] # the wavelengths are always the same
                all_spec = []
                all_err = []
            all_spec.append(star_spec)
            all_err.append(star_err)
        # weighted mean
        weights = 1.0 / (np.array(all_err)**2)
        star_spec_mean = np.sum(np.array(all_spec) * weights, axis=0) / np.sum(weights, axis=0)
        star_err_mean  = np.sqrt(1.0 / np.sum(weights, axis=0))
        # sorting by wavelengths
        star_spec_mean = star_spec_mean.transpose(2, 0, 1, 3).reshape(5,-1)
        star_err_mean = star_err_mean.transpose(2, 0, 1, 3).reshape(5,-1)
        ref_wave = ref_wave.transpose(2, 0, 1, 3).reshape(5,-1)
        sorted_indx = np.argsort(ref_wave[0,:])
        star_spec_mean = star_spec_mean[:,sorted_indx]
        star_err_mean = star_err_mean[:,sorted_indx]
        ref_wave = ref_wave[:,sorted_indx]
        Message(f'Files merged. Proceding with PSF extraction...')
        # dividing by the central row and extracting the PSF
        PSFs = []
        central_spectra = star_spec_mean[2,:]
        chunk_size = 100
        rows = np.arange(5)
        central_row_split = [central_spectra[i:i+chunk_size] for i in range(0, len(central_spectra), chunk_size)]
        stellar_split = [star_spec_mean[:, i:i+chunk_size] for i in range(0, len(star_spec_mean[0,:]), chunk_size)]
        stellar_spectra = np.zeros(star_spec_mean.shape[1])
        stellar_error = np.zeros(star_spec_mean.shape[1])
        for i, chunk in enumerate(central_row_split):
            normalised_chunk = stellar_split[i][:]/chunk
            collapsed_chunk = np.nansum(normalised_chunk, axis=1)
            spline = make_interp_spline(rows, collapsed_chunk)
            psf = spline(rows)
            psf = np.clip(psf, 0, None)
            psf /= psf.sum()
            PSFs.append(spline)
            for n in range(len(star_spec_mean[0,i*chunk_size:(i+1)*chunk_size])):
                stellar_spectra[n+chunk_size*i] = np.sum(star_spec_mean[:,n+chunk_size*i]*psf)
                stellar_error[n+chunk_size*i] = np.sqrt(np.sum(star_err_mean[:,n+chunk_size*i]**2 * psf**2))
        Message(f'PSFs extracted!')
        # saving the result
        ref_wave = ref_wave[0,:]
        star_matrix = np.column_stack((ref_wave, stellar_spectra, stellar_error))
        star_df = pd.DataFrame(star_matrix)
        star_df.to_csv(f'star.dat', index=False, header=False, sep=' ')
        Message(f'Outcome: star.dat')

    out_dir = os.path.join(OUT, date_name)
    os.makedirs(out_dir, exist_ok=True)
    shutil.move(cwd+'/product', os.path.join(out_dir, 'product'))
    shutil.move(cwd+'/star.dat', os.path.join(out_dir, 'star.dat'))
    os.chdir('/')
    shutil.rmtree(work_dir)
    Message(f'Results of {date_name} saved in {out_dir}', "i")

failed = []
for folder in folders:
    try:
        process(folder)
    except Exception:
        print(f"\n!!! {folder} FAILED, moving on to the next date !!!")
        traceback.print_exc()
        failed.append(folder)

print("\n=========================")
print(f"Done: {len(folders) - len(failed)}/{len(folders)} dates processed successfully.")
if failed:
    print("Failed folders:")
    for f in failed:
        print(f"  {f}")
# # PLANET'S SPECTRA EXTRACTION
# pipeline.remove_starlight(nod_ab='A', input_folder='correct_wavelengths_2d', lp_window_length=501, telluric_mask=(0.6, 1.6), svd_broadening_kernel=True)
# print("-------------------------\nPlanet spectra extraction\n-------------------------")
# hdu = fits.open(cwd+"/product/remove_starlight/spectra_nod_A_000.fits")
# total_rows = hdu['SPEC'].data.shape[2]
# exclude_rows = []
# for sep in pl_sep_px:
#     row = np.round((total_rows-1)/2 - sep) # check
#     rows = list(np.arange(start=row-2, stop=row+3, step=1, dtype=int))
#     exclude_rows += rows
# exclude_rows = [int(i) for i in exclude_rows] # to pass from numpy.int to int
# pipeline.remove_systematics(nod_ab='A', n_modes=5, input_folder='remove_starlight', normalize=True, exclude_rows=None)

# if(CLEAN): pipeline.clean_folder(keep_product=True)

# # MERGE AND FORMAT
# j=0
# with Task("Merging all output files", new_line=True):
#     files = [cwd+"/product/remove_systematics/"+f for f in os.listdir(cwd+"/product/remove_systematics") if os.path.isfile(os.path.join(cwd+"/product/remove_systematics", f))]
#     Message(f"Found {cstr(len(files)):c} to process", "i")
#     spec_mean = np.zeros((len(ref_wave),len(pl_sep_px)))
#     err_mean = np.zeros((len(ref_wave), len(pl_sep_px)))
#     err_mean2 = np.zeros((len(ref_wave), len(pl_sep_px)))
#     all_spec = np.zeros((len(files), len(ref_wave), len(pl_sep_px)))
#     all_err = np.zeros((len(files), len(ref_wave), len(pl_sep_px)))
#     # anti planet
#     anti_spec_mean = np.zeros((len(ref_wave),len(pl_sep_px)))
#     anti_err_mean = np.zeros((len(ref_wave), len(pl_sep_px)))
#     anti_err_mean2 = np.zeros((len(ref_wave), len(pl_sep_px)))
#     anti_all_spec = np.zeros((len(files), len(ref_wave), len(pl_sep_px)))
#     anti_all_err = np.zeros((len(files), len(ref_wave), len(pl_sep_px)))
#     # iter on files
#     for file in FancyIter(files, new_line=True):
#         hdu = fits.open(file)
#         total_rows = hdu['SPEC'].data.shape[2]
#         spec = hdu['SPEC'].data.transpose(2, 0, 1, 3).reshape(total_rows,-1)
#         spec = spec[:,sorted_indx]
#         err = hdu['ERR'].data.transpose(2, 0, 1, 3).reshape(total_rows,-1)
#         err = err[:,sorted_indx]
#         # iter on planets
#         for pl, sep in enumerate(pl_sep_px):
#             # extracting planet's PSF
#             pl_row = (total_rows-1)/2 - sep
#             pl_rows = list(np.arange(start=np.round(pl_row) - 2, stop=np.round(pl_row) + 3, step=1, dtype=int))
#             anti_pl_rows = list(np.arange(start=np.round(pl_row+2*sep) - 2, stop=np.round(pl_row+2*sep) + 3, step=1, dtype=int))
#             delta = pl_row - np.round(pl_row)
#             pl_split = [spec[:, i:i+chunk_size] for i in range(0, len(spec[0,:]), chunk_size)]
#             for i, chunk in enumerate(pl_split):
#                 psf = PSFs[i](rows-delta) # we evaluate the PSF at the pixel position
#                 psf = np.clip(psf, 0, None)
#                 psf /= psf.sum()
#                 anti_psf = PSFs[i](rows+delta) # anti
#                 anti_psf = np.clip(psf, 0, None)
#                 anti_psf /= psf.sum()
#                 for n in range(len(spec[0,i*chunk_size:(i+1)*chunk_size])):
#                     all_spec[j,n+chunk_size*i,pl] = np.sum(spec[pl_rows,n+chunk_size*i]*psf)
#                     all_err[j,n+chunk_size*i,pl] = np.sqrt(np.sum(err[pl_rows,n+chunk_size*i]**2 * psf**2))
#                     anti_all_spec[j,n+chunk_size*i,pl] = np.sum(spec[anti_pl_rows,n+chunk_size*i]*anti_psf)  # anti
#                     anti_all_err[j,n+chunk_size*i,pl] = np.sqrt(np.sum(err[anti_pl_rows,n+chunk_size*i]**2 * anti_psf**2))
#