#%%
import pandas as pd
import numpy as np
# from utils import Task, FancyIter, cstr, Message, ccf_mask, FancyArg, Velocity
# from utils.cross_correlation_tools import cross_correlate
# #from cross_correlation_mod import cross_correlate
import matplotlib.pyplot as plt
# import glob
# import re

# ##############################################
# #### Cross-correlate a single observation ####
# ##############################################


# #######################
# ## Load the spectrum ##
# #######################


# spectrum_chunk = pd.read_csv('/home/alst3712/exocomet_project/clean_data/2022-04-05_M4368/order_regions/tellurics_corrected_version_0.8.3_2022-04-05_M4368_base_preprocessed_merged.csv')
# spectrum_chunk = pd.read_csv('/home/alst3712/exocomet_project/clean_data/2023-11-01_M4318/order_regions/tellurics_corrected_version_0.8.3_2023-11-01_M4318_base_preprocessed_merged.csv')
# #spectrum_chunk = pd.read_csv('/home/alst3712/exocomet_project/clean_data/2025-09-27_M4461/order_regions/tellurics_corrected_version_0.8.3_2025-09-27_M4461_base_preprocessed_merged.csv')

T = 10
spectrum = pd.read_csv('/home/alst3712/exocomet_project/clean_data/2022-04-11_M4368/order_regions/tellurics_corrected_version_0.8.3_2022-04-11_M4368_base_preprocessed.csv')
spectrumog = pd.read_csv('/home/alst3712/exocomet_project/clean_data/2022-04-11_M4368/order_regions/tellurics_corrected_version_0.8.3_2022-04-11_M4368_base_preprocessed_merged.csv')
model = pd.read_csv(f'/home/alst3712/exocomet_project/Opacity_spectra/CO_absorptions/CO_model_{T}K_1e+17_0.9cf.csv')

# # base_chunk = '2022-04-05_M4368_2'

spectrum_sorted = (
    spectrum
    .replace([np.inf, -np.inf], np.nan)
    .dropna(subset=['wavelength', 'flux.deviation'])
    .sort_values('wavelength')
    .drop_duplicates(subset='wavelength', keep='first')
    .reset_index(drop=True)
)
plt.plot(spectrum_sorted['wavelength'], spectrum_sorted['flux.deviation'], label='tellurics corrected spectrum')
plt.plot(model['wavelength'], model['flux']-1, label='CO model')
plt.xlim(4680, 4780)
plt.ylim(-4, 1)
plt.legend()
plt.title(f'2022-04-11_M4368 spectrum and {T} K CO model')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Flux (arb. units)')
plt.show()
#%%
# # spectrum_sorted = (
# #     spectrum_chunk
# #     .replace([np.inf, -np.inf], np.nan)
# #     .dropna(subset=[f'{base_chunk}.wavelength', f'{base_chunk}.flux.no_baseline'])
# #     .sort_values(f'{base_chunk}.wavelength')
# #     .drop_duplicates(subset=f'{base_chunk}.wavelength', keep='first')
# #     .reset_index(drop=True)
# # )

# # Identify all base chunks from the column names
# base_chunks = sorted({
#     col.rsplit('.', 1)[0]
#     for col in spectrum_chunk.columns
#     if col.endswith('.wavelength')
# })

# # Create a sorted dataframe for each chunk
# spectra_sorted = {}

# for base_chunk in base_chunks:
#     spectra_sorted[base_chunk] = (
#         spectrum_chunk
#         .replace([np.inf, -np.inf], np.nan)
#         .dropna(subset=[
#             f'{base_chunk}.wavelength',
#             f'{base_chunk}.flux.no_baseline'
#         ])
#         .sort_values(f'{base_chunk}.wavelength')
#         .drop_duplicates(subset=f'{base_chunk}.wavelength', keep='first')
#         .reset_index(drop=True)
#     )

#     print(np.all(np.diff(spectra_sorted[base_chunk][f'{base_chunk}.wavelength']) > 0))
#     plt.plot(spectra_sorted[base_chunk][f'{base_chunk}.wavelength'], spectra_sorted[base_chunk][f'{base_chunk}.flux.no_baseline'])
#     plt.show()

# #%% 

# ##################
# ## Load a model ##
# ##################


# model = pd.read_csv('/home/alst3712/exocomet_project/Opacity_spectra/CO_absorptions/CO_model_400K_1e+17_0.9cf_weighted.csv')
# # model_files = glob.glob('Opacity_spectra/CO_absorptions/CO_model_*.csv')

# # models = []

# # for file in model_files:
# model = (
#     model
#     .replace([np.inf, -np.inf], np.nan)
#     .dropna(subset=['wavelength', 'flux'])
#     .sort_values('wavelength')
#     .drop_duplicates(subset='wavelength', keep='first'))
# #models.append(model)

# plt.plot(model['wavelength'][10:-2], model['flux'][10:-2])
# #print(f'Will cross-correlate with {len(models)} models')

# #%% 

# ##########################
# ## Load a shifted model ##
# ##########################

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # Speed of light in km/s
# c_kms = 299792.458

# # Load your existing model
# model = pd.read_csv(
#     '/home/alst3712/exocomet_project/Opacity_spectra/CO_absorptions/CO_model_1000K_1e+17_0.9cf_weighted.csv'
# )

# model = model.sort_values('wavelength').reset_index(drop=True)

# # Inject a known velocity
# injected_velocity = -20  # km/s

# # Create shifted spectrum
# shifted_model = model.copy()

# shifted_model['wavelength'] = (
#     model['wavelength'] * (1 + injected_velocity / c_kms)
# )

# # Save it if you want
# # shifted_model.to_csv('CO_model_1000K_1e+17_0.9cf_shifted.csv', index=False)

# #%%
# print(np.all(np.diff(spectra_sorted[base_chunk][f'{base_chunk}.wavelength']) > 0))
# print(np.all(np.diff(model['wavelength']) > 0))
# #%%

# ###################################
# ### Check which model fits best ###
# ###################################

# # for m, f in zip(models[:], model_files[:]):
# #     match = re.search(r'(\d+K_.*)$', f)
# #     extracted = match.group(1)

# #     print(extracted)    
# #     ccf = cross_correlate(m, m, velocities,
# #                             x_axis=['wavelength', 'wavelength'],
# #                             y_axis=['flux', 'f'])
# #     plt.plot(velocities, ccf)
# #     plt.title(f'Using model {extracted}')
# #     plt.show()
# #%%
# velocities = np.linspace(-300, 300, 1000)

# ###################################
# ######## Define weights ###########
# ###################################

# error = spectrum_chunk[f'{base_chunk}.error']
# model_interp = np.interp(spectrum_chunk[f'{base_chunk}.wavelength'], model['wavelength'], model['weight.smooth'])
# weight = model_interp / error**2

# model['ccf_flux'] = (
#     model['flux.no_baseline'] - model['flux.no_baseline'].mean()
# ) / model['flux.no_baseline'].std()

# shifted_model['ccf_flux'] = model['ccf_flux']

# #%%

# for bc in base_chunks:
#     spectrum_sorted = spectra_sorted[bc]
#     ccf = cross_correlate(model, spectrum_sorted, velocities, #weight=weight,
#                             x_axis=[f'wavelength', f'{bc}.wavelength'],
#                             y_axis=[f'flux.no_baseline', f'{bc}.flux.no_baseline'])

#     plt.plot(velocities, ccf)
#     #plt.xlim(-10, 10)
#     plt.xlabel('Velocity (km/s)')
#     plt.ylabel('Cross-Correlation Function (CCF)')
#     plt.title(f'Cross-Correlation for {bc}')
#     plt.show()
# #%%
# arg = ccf.argmax()
# print(velocities[arg])
# #plt.title(f'Using model {extracted}')
# plt.show()

#%%

''''''''''''''''''''''''
'''NEW CCF CODE!!!!!!'''
''''''''''''''''''''''''

import glob
import os
from plotting_params import use_my_style
use_my_style()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, median_filter, uniform_filter1d

C_KMS = 299792.458

# =============================================================================
# CONFIG
# =============================================================================

OBS_FILES = sorted(glob.glob(
    '/home/alst3712/exocomet_project/clean_data/**/*base_preprocessed_merged.csv',
    recursive=True,
))
print(len(OBS_FILES), 'observations found')


MODEL_FILES = sorted(glob.glob('/home/alst3712/exocomet_project/Opacity_spectra/CO_absorptions/CO_model_10K_*_0.9cf.csv'))

VELOCITIES = np.arange(-300, 300.01, 0.5)   # km/s
FLUX_COL = 'flux.no_baseline'               # data column used (per chunk: '<label>.<FLUX_COL>')
USE_ERROR_COLUMN = True                     # False -> use the empirical point-to-point scatter per chunk
RESOLUTION = None                           # instrument R to broaden the model to (None = model already broadened)
CLIP_SIGMA = 5.0                            # drop positive outliers (hot pixels / cosmics) above this many sigma
EXCLUDE_KMS = 40.0                          # region around the peak excluded when estimating the empirical noise


RUN_TESTS = True                            # injection / noise tests before the real run
OUT_DIR = 'ccf_output'
MODELS_TO_PLOT = None                       # list of model file names to plot, or None -> top 3 by mean per-night S/N


# =============================================================================
# LOADING
# =============================================================================
def load_observation(path):
    """Read a wide 'merged' csv -> list of raw chunk dicts (one per '<label>.wavelength' column).
    Also clean up - remove nans, duplicate wavelengths, sort by wavelength."""
    df = pd.read_csv(path)
    labels = sorted({c.rsplit('.', 1)[0] for c in df.columns if c.endswith('.wavelength')})
    chunks = []
    for lab in labels:
        wl_c, fl_c, er_c = f'{lab}.wavelength', f'{lab}.{FLUX_COL}', f'{lab}.error'
        if fl_c not in df:
            continue
        cols = [wl_c, fl_c] + ([er_c] if er_c in df else [])
        d = (df[cols].replace([np.inf, -np.inf], np.nan).dropna()
            .sort_values(wl_c).drop_duplicates(subset=wl_c))
        chunks.append(dict(
            label=lab,
            obs=lab.rsplit('_', 1)[0],           # 'date_setting' (drops the chunk number)
            wl=d[wl_c].to_numpy(float),
            flux=d[fl_c].to_numpy(float),
            err=d[er_c].to_numpy(float) if er_c in d else None,
        ))
    return chunks

def prepare_chunk(ch):
    """Noise estimate, outlier clipping, optional high-pass, rest-frame wavelength."""
    fl = ch['flux'].copy()
    dd = np.diff(fl)
    sig = 1.4826 * np.median(np.abs(dd - np.median(dd))) / np.sqrt(2)   # robust point-to-point sigma
    if not np.isfinite(sig) or sig <= 0:
        return None
    err = ch['err']
    keep = np.isfinite(err) & (err > 0) & (fl < np.median(fl) + CLIP_SIGMA * sig)  # positive outliers only: don't clip real lines
    if keep.sum() < 50:
        return None
    corr = 0
    wl = ch['wl'][keep]
    return dict(label=ch['label'], obs=ch['obs'], wl=wl, wl_corr=wl / (1 + corr / C_KMS),
                flux=fl[keep], err=err[keep], sigma_emp=sig)



def load_model(path):
    """-> (wavelength, feature) with absorption as negative dips around ~0."""
    m = pd.read_csv(path).replace([np.inf, -np.inf], np.nan)
    col = 'flux'
    m = m.dropna(subset=['wavelength', col]).sort_values('wavelength').drop_duplicates('wavelength')
    wl, f = m['wavelength'].to_numpy(float), m[col].to_numpy(float)
    f = f - 1

    if abs(f.max()) > abs(f.min()):
        print(f'  [warn] {os.path.basename(path)}: model looks like EMISSION (positive features dominate). '
            f'Flip its sign if you want absorption.')
    return wl, f


wl, f = load_model('Opacity_spectra/CO_absorptions/CO_model_1000K_1e+17_0.9cf.csv')
print(abs(f.max()), abs(f.min()))
plt.plot(wl,f)  # test load
# =============================================================================
# CORE
# =============================================================================
def chunk_ccf(wl, d, err, mw, mf, velocities=None):
    
    """
    Takes in the wavelength (wl), flux (d), and error  (err)of a single chunk,
    along with the model wavelength and flux (mw, mf).
    Return N(v), D(v) for one chunk (see module docstring)."""
    
    velocities = VELOCITIES if velocities is None else velocities
    weight = 1.0 / err ** 2
    sw = weight.sum()
    d = d - np.sum(weight * d) / sw # subtract the weighted mean (that way, the outliers stand out)
    N = np.zeros(len(velocities))
    D = np.zeros(len(velocities))
    for i, v in enumerate(velocities):
        m = np.interp(wl, mw * (1 + v / C_KMS), mf, left=0.0, right=0.0) #slide the model to the velocity v
        m = m - np.sum(weight * m) / sw #subtract the weighted mean of the model
        N[i] = np.sum(weight * d * m) #CCF numerator: multiply the data and the model (considering again the weights) and sum up all pixels 
        D[i] = np.sum(weight * m * m) #CCF denominator: useful for comparing models - same sum but with the model itself (big D means strong lines => good model 👍)
    return N, D

def model_for_chunk(mwl, mfeat, wl):
    """Crop the model to the chunk (+ velocity margin)"""
    margin = 1.5 * np.max(np.abs(VELOCITIES)) / C_KMS + 1e-4
    sel = (mwl > wl.min() * (1 - margin)) & (mwl < wl.max() * (1 + margin))
    w, f = mwl[sel], mfeat[sel]
    if len(w) < 10:
        return None
    return w, f

def run_model(chunks, model_path, inject_v=None):
    """Per-observation (N, D) sums for one model. Optional injection."""
    mwl, mfeat = load_model(model_path)
    per_obs = {}
    n_used = 0
    for ch in chunks:
        mod = model_for_chunk(mwl, mfeat, ch['wl_corr'])
        if mod is None:
            continue
        mw, mf = mod
        d = ch['flux'].copy()
        if inject_v is not None:
            d = d + np.interp(ch['wl_corr'], mw * (1 + inject_v / C_KMS), mf, left=0.0, right=0.0)
        N, D = chunk_ccf(ch['wl_corr'], d, ch['err'], mw, mf)

        #this is where chunks are combined per observation (night) - the N and D arrays are summed up for all chunks of the same night
        acc = per_obs.setdefault(ch['obs'], [np.zeros(len(VELOCITIES)), np.zeros(len(VELOCITIES))])
        acc[0] += N
        acc[1] += D
        n_used += 1
    if n_used == 0:
        print(f'  [warn] no chunk overlaps with {os.path.basename(model_path)} (wavelength units?)')
    return per_obs

def combine(per_obs):
    keys = list(per_obs)
    N = sum(per_obs[k][0] for k in keys)
    D = sum(per_obs[k][1] for k in keys)
    return N, D


def summarize(N, D, v=None, exclude=None):
    v = VELOCITIES if v is None else v
    exclude = EXCLUDE_KMS if exclude is None else exclude
    ok = D > 0
    Dsafe = np.where(ok, D, 1.0)
    A = np.where(ok, N / Dsafe, np.nan)
    snr = np.where(ok, N / np.sqrt(Dsafe), np.nan)
    if not ok.any():
        return dict(v_peak=np.nan, A=np.nan, A_err=np.nan, snr_formal=np.nan, snr_emp=np.nan), A, snr
    i = np.nanargmax(snr)
    off = (np.abs(v - v[i]) > exclude) & ok
    emp = (snr[i] - np.nanmedian(snr[off])) / np.nanstd(snr[off]) if off.sum() > 10 else np.nan
    return dict(v_peak=v[i], A=A[i], A_err=1 / np.sqrt(D[i]), snr_formal=snr[i], snr_emp=emp), A, snr


# =============================================================================
# PLOTS
# =============================================================================
def plot_nights(name, per_obs, out_dir):
    keys = sorted(per_obs)                       # keys start with the date -> chronological
    short = name.replace('CO_model_', '').replace('.csv', '')
    stats = [summarize(*per_obs[k]) for k in keys]
    snr = np.array([st[2] for st in stats])
    v_peak = np.array([st[0]['v_peak'] for st in stats])

    # 1) one panel per night, each with its own y axis (S/N in units of its own empirical noise)
    n = len(keys)
    fig, axs = plt.subplots(n, 1, figsize=(11, 1.2 + 1.3 * n), sharex=True, squeeze=False)
    axs = axs[:, 0]
    norm_snr = []

    colors = [plt.cm.Dark2(j % 8) for j in range(n)]

    for j, (ax, k, sn, vpeak) in enumerate(zip(axs, keys, snr, v_peak)):
        ipk = np.nanargmax(sn)
        sd = np.nanstd(sn[np.abs(VELOCITIES - VELOCITIES[ipk]) > EXCLUDE_KMS])
        y = sn / sd
        norm_snr.append(y)
        ax.plot(VELOCITIES, y, lw=1, color=colors[j])
        ax.text(0.01, 0.95, f'{k} -- peak: {vpeak:.1f} km/s',
                transform=ax.transAxes, fontsize=10, va='top')
        ax.axvline(16.84, color='navy', linestyle=':', lw=1.5, alpha=0.99, label='stellar RV')
        ax.set_ylabel('norm S/N', fontsize=12)
    axs[0].legend()
    axs[-1].set_xlabel('Velocity [km/s]')
    axs[0].set_title(f'Per-night CCF (each normalised to its own noise), model {short}')
    fig.tight_layout()
    fig.savefig(f'{out_dir}/nights_ccf_{short}.png', dpi=150)

    # 1b) same curves on top of each other, no date labels
    fig, ax = plt.subplots(figsize=(11, 4))
    for j, y in enumerate(norm_snr):
        ax.plot(VELOCITIES, y, lw=1, alpha=0.7, color=colors[j])
    ax.set(xlabel='Velocity [km/s]', ylabel='norm S/N',
        title=f'All nights, model {short}')
    ax.axvline(16.84, color='navy', linestyle=':', lw=1.5, alpha=0.99, label='stellar RV')
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.ylim(-5, 30)
    fig.savefig(f'{out_dir}/nights_ccf_overlay_{short}.png', dpi=150)

    # # 2) night-by-night map
    # fig, ax = plt.subplots(figsize=(11, 1.5 + 0.35 * len(keys)))
    # lim = np.nanmax(np.abs(snr))
    # im = ax.imshow(snr, aspect='auto', origin='lower', cmap='RdBu_r', vmin=-lim, vmax=lim,
    #                extent=[VELOCITIES[0], VELOCITIES[-1], -0.5, len(keys) - 0.5])
    # ax.set_yticks(range(len(keys))); ax.set_yticklabels(keys, fontsize=7)
    # ax.set(xlabel='Velocity [km/s]', title=f'S/N per night, model {short}')
    # fig.colorbar(im, ax=ax, label='S/N'); fig.tight_layout()
    # fig.savefig(f'{out_dir}/nights_map_{short}.png', dpi=150)

    # 3) variation of the peak with time
    # res = pd.DataFrame([st[0] for st in stats], index=keys)
    # fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    # x = dates if dates.notna().all() else np.arange(len(keys))
    # axs[0].plot(x, res['v_peak'], 'o-'); axs[0].set_ylabel('Peak velocity [km/s]')
    # axs[1].errorbar(x, res['A'], res['A_err'], fmt='o-'); axs[1].set_ylabel('Model scale A')
    # axs[2].plot(x, res['snr_emp'], 'o-', label='empirical'); axs[2].plot(x, res['snr_formal'], 's--', alpha=.5, label='formal')
    # axs[2].set_ylabel('S/N'); axs[2].legend()
    # for a in axs: a.grid(alpha=.3)
    # axs[0].set_title(f'Night-to-night variation, model {short}')
    # fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(f'{out_dir}/variation_{short}.png', dpi=150)
    # return res

# %%

#%%
# =============================================================================
# MAIN
# =============================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    raw = []
    for f in OBS_FILES:
        c = load_observation(f)
        print(f'{os.path.basename(f)}: {len(c)} chunks')
        raw += c
    chunks = [c for c in (prepare_chunk(r) for r in raw) if c is not None]
    obs_keys = sorted({c['obs'] for c in chunks})
    print(f'{len(chunks)} usable chunks in {len(obs_keys)} observations: {obs_keys}')
    if not chunks or not MODEL_FILES:
        raise SystemExit('No chunks or no models found - check paths.')

    mwl0, _ = load_model(MODEL_FILES[0])
    ratio = np.median(mwl0) / np.median(np.concatenate([c['wl'] for c in chunks]))
    if not 0.5 < ratio < 2:
        print(f'[warn] median model wavelength / data wavelength = {ratio:.3g}  -> unit mismatch (nm vs um vs AA)?')

    rows, results = [], {}
    for mp in MODEL_FILES:
        per_obs = run_model(chunks, mp)          # {night: [N(v), D(v)]}, chunks combined within a night only
        if not per_obs:
            continue
        name = os.path.basename(mp)
        results[name] = per_obs
        for k in sorted(per_obs):
            s, _, _ = summarize(*per_obs[k])
            rows.append(dict(model=name, obs=k, **s))

    table = pd.DataFrame(rows)
    table.to_csv(f'{OUT_DIR}/per_night_results.csv', index=False)
    pd.set_option('display.width', 200)
    print('\nEmpirical S/N per night (rows) and model (columns):')
    print(table.pivot(index='obs', columns='model', values='snr_emp').round(1).to_string())
    print('\nPeak velocity per night:')
    print(table.pivot(index='obs', columns='model', values='v_peak').round(1).to_string())

    # which model(s) to plot: ranking only, mean of per-night empirical S/N (nothing is combined)
    rank = table.groupby('model')['snr_emp'].mean().sort_values(ascending=False)
    print('\nModel ranking by mean per-night empirical S/N:\n', rank.round(2).to_string())
    for name in (MODELS_TO_PLOT or list(rank.index[:])):
        res = plot_nights(name, results[name], OUT_DIR)
        for k in sorted(results[name]):          # also save every night's CCF
            N, D = results[name][k]
            _, A, snr = summarize(N, D)
            pd.DataFrame({'velocity': VELOCITIES, 'A': A, 'snr': snr}).to_csv(
                f'{OUT_DIR}/ccf_{name.replace(".csv", "")}_{k}.csv', index=False)
    plt.show()


if __name__ == '__main__':
    main()
# %%
