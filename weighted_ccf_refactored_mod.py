"""Weighted cross-correlation analysis for CRIRES+ spectra.

The numerical calculation is available through :func:`run_weighted_ccf`.
The original command-line workflow is preserved in :func:`main`.
"""

###############
### Imports ###
###############

import argparse, pickle, os, sys, warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from scipy.optimize import curve_fit
from utils import Task, FancyIter, cstr, Message, ccf_mask, FancyArg, Velocity
from cross_correlation_mod import cross_correlate
from astropy import constants as const
import astropy.units as u

# Locate the project root without changing the caller's working directory.
# _SCRIPT_PATH = Path(__file__).resolve()
# for _candidate in (_SCRIPT_PATH.parent, *_SCRIPT_PATH.parents):
#     if (_candidate / "utils").exists():
#         sys.path.insert(0, str(_candidate))
#         break


warnings.simplefilter(action='ignore', category=FutureWarning)

_default_model = "data/MODELS/prt-flux_wl-2800nm-4800nm_species-H2O_vmr-1.00e-04_temp-1700K_rad-1.65_logg-4.18_kIR-5.0e-03_rot-20.0kms-0.81_planet-b.csv"
peak_kms = 70 # parker et al. 70km/s
#double_factor = 2 # multiplies the weight of the observation if it is in the list of double_obs


######################
##### Functions ######
######################

def Parable(x, a, b, c):
    return a*x*x + b*x + c


def compute_errors(spectra, chunk_labels):

    errors = {}

    with Task("Computing the error..."):
        for chunk_label in FancyIter(chunk_labels):

            if('K' in chunk_label and '2024-10-21_K2166' not in chunk_label):
                error = spectra[f'{chunk_label}.additional_error']
                wavelength = spectra[f'{chunk_label}.wavelength']
            else:
                error = spectra[f'{chunk_label}.error']
                wavelength = spectra[f'{chunk_label}.wavelength']

            errors[chunk_label] = {
                'error': error,
                'wavelength': wavelength
                }

    return errors


def compute_ccfs(spectra, errors, chunk_labels, chunk_contrast_models, velocities, WEIGHT, SUBTRACT=False, autocorrelation=False):

    w_dict = {}
    chunk_windows = []

    chunk_labels = sorted(
        chunk_labels,
        key=lambda x: (
            x.split('_')[1],
            x.split('_')[0],
            int(x.split('_')[2])
        )
    )

    with Task("Computing the CCFs..."):

        for chunk_label in FancyIter(chunk_labels):
            with FancyIter.silence(): # avoid print statements for function 'cross_correlate'
                # only for subtraction
                if SUBTRACT:
                    try:
                        model_contrast_interp = np.interp(spectra[f'{chunk_label}.wavelength'], chunk_contrast_models[chunk_label]['wavelength'], chunk_contrast_models[chunk_label]['sub'])
                    except KeyError:
                        print(f"[ WARNING ]: {chunk_label} has no model")
                    if('2024-10-23_H1567' in chunk_label): spectra[f'{chunk_label}.flux'] = spectra[f'{chunk_label}.flux'] - 19.75 * model_contrast_interp
                    elif('2024-10-24_H1567' in chunk_label): spectra[f'{chunk_label}.flux'] = spectra[f'{chunk_label}.flux'] - 42.50 * model_contrast_interp
                    elif('2024-11-21_H1567' in chunk_label): spectra[f'{chunk_label}.flux'] = spectra[f'{chunk_label}.flux'] - 24.17 * model_contrast_interp
                    elif('2024-11-22_H1567' in chunk_label): spectra[f'{chunk_label}.flux'] = spectra[f'{chunk_label}.flux'] - 25.83 * model_contrast_interp
                    elif('2024-11-28_H1567' in chunk_label): spectra[f'{chunk_label}.flux'] = spectra[f'{chunk_label}.flux'] - 18.75 * model_contrast_interp
                    elif('2024-12-03_H1567' in chunk_label): spectra[f'{chunk_label}.flux'] = spectra[f'{chunk_label}.flux'] - 41.67 * model_contrast_interp
                    elif('2025-02-01_H1567' in chunk_label): spectra[f'{chunk_label}.flux'] = spectra[f'{chunk_label}.flux'] - 8.33 * model_contrast_interp
                    '''elif('2024-10-21_K2166' in chunk_label): spectra[f'{chunk_label}.flux'] = spectra[f'{chunk_label}.flux'] - 5.82 * model_contrast_interp
                    elif('L' in chunk_label): spectra[f'{chunk_label}.flux'] = spectra[f'{chunk_label}.flux'] - 0.64 * model_contrast_interp
                    elif('M' in chunk_label): spectra[f'{chunk_label}.flux'] = spectra[f'{chunk_label}.flux'] - 0.69 * model_contrast_interp
                    else: spectra[f'{chunk_label}.flux'] = spectra[f'{chunk_label}.flux'] - 0.08 * model_contrast_interp'''
                    #spectra[f'{chunk_label}.flux'] = spectra[f'{chunk_label}.flux'] - SUBTRACT[0] * model_contrast_interp + SUBTRACT[1]
                # get the spectrum
                chunk = spectra[[
                    f'{chunk_label}.wavelength',
                    f'{chunk_label}.flux.no_baseline', # not used
                    f'{chunk_label}.flux'
                ]].dropna()
                chunk_windows.append((chunk[f'{chunk_label}.wavelength'].min(), chunk[f'{chunk_label}.wavelength'].max()))
                # get the model at the corresponding wavelengths and the errors
                try: model = chunk_contrast_models[chunk_label]
                except KeyError:
                    print(f"[ WARNING ] No model for {chunk_label}")
                    continue
                model = model[model['wavelength'].between(*chunk_windows[-1])]
                if(model.empty): print(f'[ WARNING ]: no model avaible for chunk window {chunk_windows[-1]}')
                error = errors[chunk_label]['error']
                error_wave = errors[chunk_label]['wavelength']
                error = np.interp(chunk[f'{chunk_label}.wavelength'], error_wave, error)
                if((chunk[f'{chunk_label}.wavelength'].min()>4000) & (np.std(chunk[f"{chunk_label}.flux"])<1e-4)):
                    # for CO2 absorptions and ultra-noisy chunks: no CCF is computed, the chunk is discarded below (weight 0, NaN CCF).
                    # (Before, 'ccf' and 'model_mask_interp' were left unset here: UnboundLocalError if this was the first chunk,
                    # and silently the PREVIOUS chunk's values otherwise.)
                    info_weights = 0
                    ccf = np.full(len(velocities), np.nan)
                    model_mask_interp = np.zeros(len(chunk))
                    print(f"[ WARNING ]: {chunk_label} discarded (wavelength > 4000 nm and flux std = {np.std(chunk[f'{chunk_label}.flux']):.1e} < 1e-4)")
                else:
                    model_mask_interp = np.interp(chunk[f'{chunk_label}.wavelength'], model['wavelength'], model['weight.mask'])
                    model_interp = np.interp(chunk[f'{chunk_label}.wavelength'], model['wavelength'], model['weight.smooth'])

                    if(WEIGHT):
                        if("K" in chunk_label): error[error<9e-6] = np.nan
                        else: error[error<5e-6] = np.nan
                        weight = model_interp if "H" in chunk_label else (model_interp / error**2)
                        info_weights = np.nansum(model_mask_interp)
                        # cross-correlating
                        if(autocorrelation):
                            chunk[f'{chunk_label}.flux'] = np.interp(chunk[f'{chunk_label}.wavelength'], model['wavelength'], model['contrast.no_baseline'])
                            weight = np.ones(len(weight))
                        chunk.loc[:,f'{chunk_label}.flux'] -= np.mean(chunk[f'{chunk_label}.flux'])
                        model.loc[:,'contrast.no_baseline'] -= np.mean(model['contrast.no_baseline'])
                        ccf = cross_correlate(
                                chunk,
                                model,
                                velocities,
                                mask = ccf_mask(chunk[f'{chunk_label}.flux']),
                                weight = weight,
                                x_axis=[f'{chunk_label}.wavelength','wavelength'],
                                y_axis=[f'{chunk_label}.flux','contrast.no_baseline'],
                                center = False
                            )

                    else:
                        if(autocorrelation):
                            chunk[f'{chunk_label}.flux'] = np.interp(chunk[f'{chunk_label}.wavelength'], model['wavelength'], model['contrast.no_baseline'])
                            weight = np.ones(len(weight))
                        info_weights = np.nansum(model_mask_interp)
                        chunk.loc[:,f'{chunk_label}.flux'] -= np.mean(chunk[f'{chunk_label}.flux'])
                        model.loc[:,'contrast.no_baseline'] -= np.mean(model['contrast.no_baseline'])
                        ccf = cross_correlate(
                                chunk,
                                model,
                                velocities,
                                mask = ccf_mask(chunk[f'{chunk_label}.flux']),
                                weight = model_interp,
                                x_axis=[f'{chunk_label}.wavelength','wavelength'],
                                y_axis=[f'{chunk_label}.flux','contrast.no_baseline'],
                                center = False
                            )


                # discard the NaN and save the results
                if np.isnan(ccf).any():
                    print(f'[ WARNING ]: {chunk_label} generates NaN CCF')
                    w_dict[f"{chunk_label}.weight"] = 0
                    w_dict[f"{chunk_label}.ccfs"] = ccf
                else:
                    if info_weights == np.inf:
                        w_dict[f"{chunk_label}.weight"] = 0
                    if 'H1567_6' in chunk_label:
                        print(f"[ WARNING ] {chunk_label}'s weight set to zero")
                        w_dict[f"{chunk_label}.weight"] = 0
                    else:
                        w_dict[f"{chunk_label}.weight"] = info_weights
                    w_dict[f"{chunk_label}.ccfs"] = ccf

                try: w_dict[f"{chunk_label}.noise"] = np.nanmedian(error[model_mask_interp.astype(bool)]**2) if len(error[model_mask_interp.astype(bool)])>0 else 1
                except IndexError: w_dict[f"{chunk_label}.noise"] = np.nanmedian(error**2) if len(error)>0 else 1

    return w_dict

def run_weighted_ccf(
    spectra,
    model,
    *,
    planet,
    species,
    vmr,
    exotic_mol,
    model_path,
    rv_min,
    velocities,
    err_weight=True,
    anti_snr=True,
    subtract=None,
    anti_spectra=None,
    reference_weights=None,
):
    """Compute the weighted CCF for already-loaded spectra and model.

    The existing per-chunk CCF calculation and two-stage weighting scheme are
    retained. File loading, result persistence, and plotting are handled by
    :func:`main` and its helpers.

    Parameters are deliberately explicit so this function can be imported and
    called from a row-parallel wrapper without relying on CLI globals.

    Returns
    -------
    dict
        Always includes ``ccf_df``, scalar CCF S/N values, the CCF peak,
        estimated noise scale, possibly modified spectra, and a compact
        per-setting weight summary. 
    """
    # Work on a private DataFrame because the existing subtraction branch
    # modifies spectra in place.
    spectra = spectra.copy(deep=True)
    if anti_spectra is not None and anti_spectra is not spectra:
        anti_spectra = anti_spectra.copy(deep=True)
    subtract = [] if subtract is None else subtract
    species_name = "-".join(species)

    peak_in = pd.Series(velocities).between(-peak_kms, peak_kms)
    peak_out = pd.Series(velocities).between(-peak_kms - 30, peak_kms + 30)

    chunk_labels = [col.replace('.flux', '') for col in spectra.columns if col.endswith('.flux')]
    chunk_labels = sorted(chunk_labels, key=lambda x: x.split('_')[0])
    #observations = sorted(set([label[:-2] for label in chunk_labels]))

    #to_skip = {}
    for chunk_label in chunk_labels:
        date = chunk_label.split('_')[0]
        #vel_dict = Velocity.get(planet, date)

        # if abs(vel_dict["vrad_planet"] / vel_dict["vrad_planet_amplitude"]) < rv_min:
        #     to_skip[chunk_label[:-2]] = vel_dict["vrad_planet"] # do not use the last two characters, which are '_{chunk_number}'

    #rv_min_str = f'{rv_min:.0%}'
    #k_amp = f'{vel_dict["vrad_planet_amplitude"]:.1f}'

    # if len(to_skip) == 0:
    #     Message(f"No observations to skip based on separation criteria (at least {cstr(rv_min_str):c} of {cstr(k_amp):c} km/s).", "#")
    # else:
    #     Message(f"The following observations will be skipped based on separation criteria (at least {cstr(rv_min_str):c} of {cstr(k_amp):c} km/s):", "?")
    #     with Message.tab():
    #         for chunk_label, velocity in to_skip.items():
    #             velocity_as_str = f"{velocity:.1f}"
    #             Message.print(f"{chunk_label} ({cstr(velocity_as_str):y} km/s)")

    # chunk_labels = [chunk_label for chunk_label in chunk_labels if not chunk_label[:-2] in to_skip]


    ##################################
    ### Transform flux to contrast ###
    ##################################


    COMPUTE_WITH_CONTRAST = False #'stellar_flux' in model.columns

    if not COMPUTE_WITH_CONTRAST:
        Message(f"Model {cstr(os.path.basename(model_path)):r} does not contain a stellar flux", "!")
        Message(f"In order do compute the CCF with contrast and not flux, recompute your model with '{cstr('create_model_prt.py')}' and provide a valid star file", "!")
        Message(f"CCF will be computed with flux only...", '!')
    else:
        Message("Found stellar data. Contrast will be applied and shifts are taken into account.", '#')


    chunk_contrast_models = {} # we use the same model for all, but the stellar spectrum is different for each chunk (shifted depnding on the planet rad_vel, depending on date)
    model_baseline = model['flux'] - model['flux.no_baseline']
    # The contrast model only depends on the observation date (planet velocity), not on the chunk. It is built ONCE per date and the
    # same object is shared by all the chunks of that date (it is never modified: compute_ccfs works on a cropped copy).
    # Before, every chunk had its own full-resolution copy (~100 MB each for a 1.8 M-point model).
    models_by_date = {}
    with Task("Transforming flux to contrast", new_line=False):
        for i, chunk_label in enumerate(chunk_labels):

            model_key = chunk_label.split('_')[0] if COMPUTE_WITH_CONTRAST else None
            if model_key in models_by_date:
                chunk_contrast_models[chunk_label] = models_by_date[model_key]
                continue

            if not COMPUTE_WITH_CONTRAST:
                chunk_contrast_models[chunk_label] = pd.DataFrame({
                    'wavelength':model['wavelength'],
                    'contrast.no_baseline':model['flux.no_baseline'], # /!\ this is no contrast, but contrast cannot be used!
                    'weight.mask':model['weight.mask'],
                    'weight.smooth':model['weight.smooth']
                })
                models_by_date[model_key] = chunk_contrast_models[chunk_label]
                continue

            date = chunk_label.split('_')[0]
            planet_vel = Velocity.get(planet, date)['vrad_planet'] # we need that to know how to shift the stellar spectrum relative to the planet
            # this was the relative radial velocity between planet and star

            stellar_spectrum = model[
                ['wavelength', 'stellar_flux', 'stellar_flux.baseline']
            ].copy(deep=True)

            # let's apply the correct doppler shift
            z = planet_vel / const.c.to(u.km/u.s).value
            stellar_spectrum['wavelength'] = stellar_spectrum['wavelength'] * (1 + z)**(-1) # doing this changes absolutely nothing, the H absorption peaks in the stellar spectrum are wide enough to not be affected by the doppler shift
            # justification for doppler shift (for beta pic b):
            # => z > 0 because picB goes away
            # => here we are in ref of picB, so * gets closer => blue shift
            # => wl must decrease, but 1+z>1 so we must divide by it #
            stellar_spectrum = pd.DataFrame({
                'wavelength':model['wavelength'], # let's map it back on the initial model wavelengths
                'stellar_flux':np.interp(model['wavelength'], stellar_spectrum['wavelength'], stellar_spectrum['stellar_flux']),
                'stellar_flux.baseline':np.interp(model['wavelength'], stellar_spectrum['wavelength'], stellar_spectrum['stellar_flux.baseline'])
            })

            contrast = model['flux'] / stellar_spectrum['stellar_flux']
            contrast_no_baseline = contrast - model_baseline / stellar_spectrum['stellar_flux'] # the second part here is very smooth, don't worry about it!
            contrast_no_baseline_fake = contrast - model_baseline / stellar_spectrum['stellar_flux.baseline'] # this removes effectively the baseline but not the features linked to stllar absorption peaks, who help improve the SNR

            chunk_contrast_models[chunk_label] = pd.DataFrame({
                'wavelength':model['wavelength'],
                'contrast.no_baseline': contrast, # /!\ contrast_no_baseline lowers significantly the SNR => contrast_no_baseline_fake is almost identical to contrast in terms of SNR, so its really the stellar features that matter
                'weight.mask':model['weight.mask'],
                'weight.smooth':model['weight.smooth'],
                'weight': model['weight'],
                'sub': contrast_no_baseline_fake,
                'pseudo_contrast': pd.Series(contrast.to_numpy()[::-1]) # is used to remove systematics
            }) # only flux of model is used. If instead use flux.no_baseline, please modify run_mcmc.py in get_broadened_model()
            models_by_date[model_key] = chunk_contrast_models[chunk_label]


    #with open("model_dict.pkl", "wb") as f:
    #    pickle.dump(chunk_contrast_models, f)


    ###############################
    ### Compute errors and CCFs ###
    ###############################

    # planet
    errors = compute_errors(spectra, chunk_labels)
    w_dict = compute_ccfs(spectra, errors, chunk_labels, chunk_contrast_models, velocities, err_weight, subtract)
    # anti planet
    anti_w_dict = None
    if anti_snr and not(exotic_mol) and anti_spectra is not None:
        anti_chunk_labels = [col.replace('.flux', '') for col in anti_spectra.columns if col.endswith('.flux')]
        anti_errors = compute_errors(anti_spectra, anti_chunk_labels)
        anti_w_dict = compute_ccfs(anti_spectra, anti_errors, anti_chunk_labels, chunk_contrast_models, velocities, err_weight)
    # autocorrelation
    elif(exotic_mol):
        Message("Autocorrelation...", "#")
        anti_w_dict = compute_ccfs(spectra, errors, chunk_labels, chunk_contrast_models, velocities, err_weight, autocorrelation=True)
    #    Message("Trying as before, test!!!", "#")
    #    anti_chunk_labels = [col.replace('.flux', '') for col in anti_spectra.columns if col.endswith('.flux')]
    #    anti_errors = compute_errors(anti_spectra, anti_chunk_labels)
    #    anti_w_dict = compute_ccfs(anti_spectra, anti_errors, anti_chunk_labels, chunk_contrast_models, velocities, err_weight)



    ################################
    ### Compute the weighted CCF ###
    ################################


    with Task("Computing the weights..."):
        spectral_settings = []
        for chunk_label in chunk_labels: spectral_settings.append(chunk_label.split('_')[0] + '_'  + chunk_label.split('_')[1])
        spectral_settings = set(spectral_settings)

        # if we have several spectral settings we compute the SNR of each one of them
        if(len(spectral_settings)>1):
            Message(f"{cstr(len(spectral_settings)):g} observation blocks, computing their SNR...", "#")
            for spectral_setting in FancyIter(spectral_settings):
                ccf = np.zeros_like(velocities)
                keys = [key for key in w_dict if spectral_setting in key]
                # let's compute the weighted CCF of each chunk within a spectral setting
                ccfs = np.array([w_dict[k] for k in keys if 'ccfs' in k])
                weights = np.array([w_dict[k] for k in keys if 'weight' in k])
                noise = np.array([w_dict[k] for k in keys if 'noise' in k])
                if(weights.sum()==0):
                    w_dict[f'{spectral_setting}.snr'] = 0
                    w_dict[f'{spectral_setting}.chunk_ccfs'] = ccf.copy()
                    if('H2O' in species): w_dict[f'{spectral_setting}.correction_factor'] = 0
                    if(anti_snr):
                        anti_w_dict[f'{spectral_setting}.snr'] = 0
                        anti_w_dict[f'{spectral_setting}.chunk_ccfs'] = ccf.copy()
                    continue
                for w, ccf_chunk in zip(weights/weights.sum(), ccfs):
                    if(np.isnan(ccf_chunk[0])): continue
                    ccf += ccf_chunk * w
                if(anti_snr):
                    anti_keys = [key for key in anti_w_dict if spectral_setting in key]
                    anti_ccf = np.zeros_like(velocities)
                    anti_ccfs = np.array([anti_w_dict[k] for k in anti_keys if 'ccfs' in k])
                    anti_weights = np.array([anti_w_dict[k] for k in anti_keys if 'weight' in k])
                    for w, ccf_chunk in zip(anti_weights/anti_weights.sum(), anti_ccfs):
                        if(np.isnan(ccf_chunk[0]) or anti_weights.sum()==0): continue
                        anti_ccf += ccf_chunk * w
                # and now we merge the spectral settings! (using SNR ecc...)
                try:
                    popt, pcov = curve_fit(Parable, velocities[~peak_out], ccf[~peak_out])
                    noise_est = Parable(velocities, *popt)
                except RuntimeError:
                    Message("Failed to fit a parable to the CCF's tails", "i")
                    noise_est = np.zeros_like(velocities)
                ccf_clean = ccf - noise_est

                # when looking for expected molecules (e.g. H2O, CO), the merge is based on the spectral setting's SNR. But when we look for 'exotic' species, we can't do that because the SNR might be too low
                if(not(exotic_mol)):
                    if(anti_snr):
                        try:
                            popt, pcov = curve_fit(Parable, velocities[~peak_out], anti_ccf[~peak_out])
                            noise_est = Parable(velocities, *popt)
                        except RuntimeError:
                            Message("Failed to fit a parable to the CCF's tails", "!")
                            noise_est = np.zeros_like(velocities)
                        anti_ccf_clean = anti_ccf - noise_est
                        snr = ccf_clean[peak_in].max()/anti_ccf_clean[~peak_out].std()
                    else: snr = ccf_clean[peak_in].max() / ccf_clean[~peak_out].std()
                # exotic molecules
                else:
                    snr = reference_weights.loc[f"{spectral_setting}.correction_factor"].to_numpy() * (np.nansum(weights/noise))  # correction factor
                    if(anti_snr):
                        try:
                            popt, pcov = curve_fit(Parable, velocities[~peak_out], anti_ccf[~peak_out])
                            noise_est = Parable(velocities, *popt)
                        except RuntimeError:
                            Message("Failed to fit a parable to the CCF's tails", "!")
                            noise_est = np.zeros_like(velocities)
                        anti_ccf_clean = anti_ccf - noise_est
    #                    snr = np.nansum(weights)/anti_ccf_clean[~peak_out].var()
    #                auto_ccf = anti_ccf_clean - np.mean(anti_ccf_clean)
    #                factor = np.dot(ccf_clean, auto_ccf) / np.dot(auto_ccf, auto_ccf)
    #                noise_ccf = ccf_clean - factor * auto_ccf
    #                snr = anti_ccf_clean[peak_in].max() / noise_ccf[~peak_out].std() #ccf_clean[~peak_out].std()  #np.nanstd(noise)

                # when we subtract a specie we then want to check if the subtraction was effective (SNR~0), but to do that we set the weight to what obtained with the normal case
                if(len(subtract)>0):
                    w_dict[f'{spectral_setting}.snr'] = reference_weights.loc[f"{spectral_setting}.snr"].to_numpy()
                    w_dict[f'{spectral_setting}.chunk_ccfs'] = ccf_clean.copy()
                    if(anti_snr):
                        anti_w_dict[f'{spectral_setting}.snr'] = reference_weights.loc[f"{spectral_setting}.snr"].to_numpy()
                        anti_w_dict[f'{spectral_setting}.chunk_ccfs'] = anti_ccf_clean.copy()
                # normal behaviour
                else:
                    if not np.isfinite(snr) or snr < 0:
                        w_dict[f'{spectral_setting}.snr'] = 0
                        w_dict[f'{spectral_setting}.chunk_ccfs'] = ccf_clean.copy()
                        if('H2O' in species): w_dict[f'{spectral_setting}.correction_factor'] = 0
                        if(anti_snr):
                            anti_w_dict[f'{spectral_setting}.snr'] = 0
                            anti_w_dict[f'{spectral_setting}.chunk_ccfs'] = anti_ccf_clean.copy()
                    else:
                        w_dict[f'{spectral_setting}.snr'] = snr
                        w_dict[f'{spectral_setting}.chunk_ccfs'] = ccf_clean.copy()
                        if('H2O' in species): w_dict[f'{spectral_setting}.correction_factor'] = snr / (np.nansum(weights/noise))
                        if(anti_snr):
                            anti_w_dict[f'{spectral_setting}.snr'] = snr
                            anti_w_dict[f'{spectral_setting}.chunk_ccfs'] = anti_ccf_clean.copy()
                weight_keys = [key for key in keys if 'weight' in key]
                for k, w in zip(weight_keys, weights): w_dict[k] = w/np.sum(weights) # for plotting reasons
            # the final ccf accounts for the SNR of each spectral setting
            ccf = np.zeros_like(velocities)
            total_ccf = np.array([w_dict[k] for k in w_dict.keys() if 'chunk_ccfs' in k])
            snr = np.array([
                np.asarray(w_dict[k]).item() if np.asarray(w_dict[k]).size == 1 else w_dict[k]
                for k in w_dict
                if 'snr' in k
            ])
            snr_keys = [key for key in w_dict.keys() if 'snr' in key]
            for l, e in zip(snr_keys, snr): print(f"{l}: {e}") ## !!!
            # quitting if the SNR sum is equal to 0
            if(snr.sum()==0):
                raise RuntimeError("Every observation produced SNR <= 0.")
            # ccf calculations
            for w, ccf_chunk in zip(snr**2, total_ccf):    # instead of snr/snr.sum()
                if(np.isnan(ccf_chunk[0])): continue
                ccf += w * ccf_chunk
            if(anti_snr):
                anti_ccf = np.zeros_like(velocities)
                anti_total_ccf = np.array([anti_w_dict[k] for k in anti_w_dict.keys() if 'chunk_ccfs' in k])
                #snr = np.array([anti_w_dict[k] for k in anti_w_dict.keys() if 'snr' in k])
                snr = np.array([
                    np.asarray(anti_w_dict[k]).item() if np.asarray(anti_w_dict[k]).size == 1 else anti_w_dict[k]
                    for k in anti_w_dict
                    if 'snr' in k
                ])
                for w, ccf_chunk in zip(snr**2, anti_total_ccf): # instead of snr/snr.sum()
                    if(np.isnan(ccf_chunk[0])): continue
                    anti_ccf += w * ccf_chunk

        # if we only have one we follow the normal procedure
        else:
            ccf = np.zeros_like(velocities)
            Message(f"Only {cstr(1):g} observation block, following standard weighting...", "#")
            ccfs = np.array([w_dict[k] for k in w_dict.keys() if 'ccfs' in k])
            weights = np.array([w_dict[k] for k in w_dict.keys() if 'weight' in k])
            for k, w in zip(w_dict.keys(), weights): w_dict[k] = w/np.sum(weights) # for plotting reasons
            for w, ccf_chunk in zip(weights/weights.sum(), ccfs):
                if(np.isnan(ccf_chunk[0])): continue
                ccf += w * ccf_chunk
            try:
                popt, pcov = curve_fit(Parable, velocities[~peak_out], ccf[~peak_out])
                noise_est = Parable(velocities, *popt)
            except RuntimeError:
                Message("Failed to fit a parable to the CCF's tails", "i")
                noise_est = np.zeros_like(velocities)
            ccf = ccf - noise_est
            if(anti_snr):
                anti_ccf = np.zeros_like(velocities)
                anti_total_ccf = np.array([anti_w_dict[k] for k in anti_w_dict.keys() if 'ccfs' in k])
                for w, ccf_chunk in zip(weights/weights.sum(), anti_total_ccf):
                    if(np.isnan(ccf_chunk[0])): continue
                    anti_ccf += w * ccf_chunk
                try:
                    popt, pcov = curve_fit(Parable, velocities[~peak_out], anti_ccf[~peak_out])
                    noise_est = Parable(velocities, *popt)
                except RuntimeError:
                    Message("Failed to fit a parable to the CCF's tails", "i")
                    noise_est = np.zeros_like(velocities)
                anti_ccf = anti_ccf - noise_est


    if(anti_snr):
        ccf_df = pd.DataFrame({
            'velocity': velocities,
            'ccf': ccf,
            'anti_ccf': anti_ccf
        })
    else:
        ccf_df = pd.DataFrame({
            'velocity': velocities,
            'ccf': ccf,
        })

    # let's display SN ratio of the resulting CCF
    ccf_snr = None
    anti_ccf_snr = None
    ccf_peak = None
    noise_sigma = None
    ccf_snr_str = None
    anti_ccf_snr_str = None
    auto_ccf = None

    if((not anti_snr)):
        if len(ccf_df[~peak_out]) > 0 and len(ccf_df[peak_in]) > 0:
            ccf_peak = ccf_df['ccf'][peak_in].max() if ccf_df['ccf'][peak_in].max()>np.abs(ccf_df['ccf'][peak_in].min()) else ccf_df['ccf'][peak_in].min()
            ccf_pk_str = f"{ccf_peak:.2e}"
            noise_sigma = ccf_df['ccf'][~peak_out].std()
            ccf_snr = ccf_peak / noise_sigma
            ccf_snr_str = f"{ccf_snr:.2f}"
        else: pass  # The caller reports that the velocity grid is too narrow.

    else:
        ccf_peak = ccf_df['ccf'][peak_in].max() if ccf_df['ccf'][peak_in].max()>np.abs(ccf_df['ccf'][peak_in].min()) else ccf_df['ccf'][peak_in].min()
        ccf_pk_str = f"{ccf_peak:.2e}"
        if(exotic_mol):
            ccf = ccf_df['ccf'].to_numpy() - np.mean(ccf_df['ccf'].to_numpy()[~peak_out])
            anti = anti_ccf - np.mean(anti_ccf[~peak_out])
            print(ccf_df['ccf'].to_numpy() - ccf)
            factor = np.max(ccf[peak_in])/np.max(anti[peak_in])
            auto_ccf = anti_ccf
            #factor = np.dot(ccf, anti) / np.dot(anti, anti)
            anti_ccf = ccf - factor * anti
            ccf_df['anti_ccf'] = anti_ccf
            ccf_df['auto_ccf'] = auto_ccf * factor
            noise_sigma = anti_ccf[~peak_out].std()
            ccf_snr = ccf_peak / noise_sigma
            ccf_snr_str = f"{ccf_snr:.2f}"
        else:
            noise_sigma = anti_ccf[~peak_out].std()
            ccf_snr = ccf_peak / noise_sigma
            ccf_snr_str = f"{ccf_snr:.2f}"
        anti_ccf_snr = anti_ccf[peak_in].max() / anti_ccf[~peak_out].std()
        anti_ccf_snr_str = f"{anti_ccf_snr:.2f}"



    # Compact per-setting summary, also used by the original CLI output writer.
    summary_keys = [
        key for key in w_dict
        if key.endswith('.snr') or key.endswith('.correction_factor')
    ]
    weight_summary = pd.DataFrame(index=summary_keys)
    weight_summary['value'] = [w_dict[key] for key in summary_keys]


    result = {
        "ccf_df": ccf_df,
        "velocity": velocities.copy(),
        "ccf": ccf_df["ccf"].to_numpy(copy=True),
        "ccf_snr": ccf_snr,
        "peak_snr": ccf_snr,  # Optional backward-compatible alias
        "ccf_peak": ccf_peak,
        "noise_sigma": noise_sigma,
        "anti_ccf": anti_ccf.copy() if anti_snr else None,
        "anti_ccf_snr": anti_ccf_snr,
        "spectra": spectra,
        "weight_summary": weight_summary,
        "vmr": vmr,
        "species_name": species_name,
        "w_dict": w_dict,
        "anti_w_dict": anti_w_dict,
    }
    if 'auto_ccf' in locals():
        result["auto_ccf"] = auto_ccf

    return result


def main():
    """Run the original command-line workflow, including saving and plots."""
    with FancyArg.create("Compute the weighted cross-correlation function of the different precomputed chunks"):

        # files
        FancyArg(
            "f", type=str, default="data/CRIRES/CRIRES_b", help="File or folder (not just the name, the entire path). If file, must be a .csv file, output of merge_observations.py. If folder, all the available .csv files at the root of the folder (where the output of merge_observations.py are) will be processed."
        )
        FancyArg(
            "model", default = _default_model, help = f"The path to the precomputed model to cross-correlate with. Last letter of model filename fixes the planet."
        )
        FancyArg(
            "planet", default="bet Pic b", type=str, help="Planet, necessary to compute velocity corrections. Give the EXACT planet name as used by NASA Exoplanet Archive (case and space sensitive). Type 'None' if you don't want to compute velocity corrections"
        )
        FancyArg(
            "min_rv", type=float, default=0.5, help="Between 0 and 1, minimum ratio between radial velocity and maximum radial velocity on the orbit (K factor). If this ratio is 0, it corresponds to being in front or behind the star, so very low separation."
        )
        # ccf computation
        FancyArg(
            "vel_linspace", type=list, default=["-300", "300", "601"], help="List of integers that will be passed to np.linspace to create the velocity array (in km/s)."
        )
        # additional weighting scheme
        FancyArg(
            "err_weight", type=bool, default=True, help="Additional weighting scheme which can be added to the chunks information-based one, based on the noise of the chunk."
        )
        # different SNR evaluation
        FancyArg(
            "anti_SNR", type=bool, default=True, help="The evaluation of the SNR is performed using the signal from the other side of the star and not using the tails of the CCF."
        )
        # for molecules subtraction
        FancyArg(
            "subtract", type=list, default=[], help="Fill in True ONLY if you want to subtract of a species and then cross-correlate to look for the model's normalisation factor."
        )
        # display
        FancyArg(
            "show", type=bool, default=True, help="Show the plots during the process."
        )

    # get args and check validity
    file_or_folder, MODEL, PLANET, RV_MIN, VEL_LINSPACE, WEIGHT, ANTI_SNR, SUBTRACT, SHOW = FancyArg.get()

    assert os.path.exists(file_or_folder), f"Path {cstr(file_or_folder):r} not found!"
    assert os.path.isfile(MODEL), f"Model file {cstr(MODEL):r} not found!"
    assert MODEL.endswith('.csv'), f"Model file must be a {cstr('.csv'):r} file, not {cstr(os.path.basename(MODEL)):r}"

    if not os.path.isdir(file_or_folder):
        assert file_or_folder.endswith('.csv'), f"File must be a {cstr('.csv'):r} file (output of merge_observations.py), not {cstr(os.path.basename(file_or_folder)):r}"

    assert 0 <= RV_MIN <= 1, f"RV_MIN must be between 0 and 1, not {cstr(RV_MIN):r}"


    if isinstance(PLANET, str) and PLANET.strip().lower() == "none":
        PLANET = None
    else:
        PLANET = PLANET.strip()
    Message(f"Planet identified as: {cstr(PLANET):g}", "#")

    with Task("Creating the velocity array", new_line=True):
        linspace_as_str = f"np.linspace({','.join(VEL_LINSPACE)})"
        Message(f"Initializing with: {cstr(linspace_as_str):c}", "i")
        velocities = np.linspace(*[int(x) for x in VEL_LINSPACE])
        peak_in = pd.Series(velocities).between(-peak_kms, peak_kms)
        peak_out = pd.Series(velocities).between(-peak_kms-30, peak_kms+30)

    vmr = MODEL.split("/")[-1].split("_")[3]
    vmr = vmr.split("-")[1] + "-" + vmr.split("-")[2]
    SPECIES = MODEL.split("/")[-1].split("_")[2].split("-")[1:]
    EXOTIC_MOL = not(('H2O' in SPECIES) or ('CO' in SPECIES))
    #EXOTIC_MOL = True
    species_name = "-".join(SPECIES)

    SNR_sub = None


    if(len(SUBTRACT)>0):
        SUBTRACT = [float(sub) for sub in SUBTRACT]
        weight_path = os.path.dirname(file_or_folder)+f"/weights/w_dict_ref_H2O.csv"
        assert os.path.exists(weight_path), f"File {weight_path} doesn't exist. Please generate the file."
        SNR_sub = pd.read_csv(weight_path, index_col=0, sep=' ')

    if(EXOTIC_MOL):
        weight_path = os.path.dirname(file_or_folder)+f"/weights/w_dict_ref_H2O.csv"
        assert os.path.exists(weight_path), f"File {weight_path} doesn't exist. Please generate the file."
        SNR_sub = pd.read_csv(weight_path, index_col=0, sep=' ')

    #################################################################################################################################################################################################################################

    #####################
    ### Read the file ###
    #####################

    with Task("Reading files", new_line=False):
        spectra = pd.read_csv(file_or_folder)

        anti_spectra = None
        if ANTI_SNR:
            anti_file = (
                file_or_folder.replace('_anti', '')
                if '_anti_' in file_or_folder
                else file_or_folder
                )

            if os.path.exists(anti_file): anti_spectra = pd.read_csv(anti_file)
            else:
                Message(f"{anti_file} not found!", "!")
                quit()
        model = pd.read_csv(MODEL)

    # Execute the scientific calculation. Diagnostics are retained here because
    # the legacy CLI plotting routines consume the intermediate CCF dictionaries.
    result = run_weighted_ccf(
        spectra=spectra,
        model=model,
        planet=PLANET,
        species=SPECIES,
        vmr=vmr,
        exotic_mol=EXOTIC_MOL,
        model_path=MODEL,
        rv_min=RV_MIN,
        velocities=velocities,
        err_weight=WEIGHT,
        anti_snr=ANTI_SNR,
        subtract=SUBTRACT,
        anti_spectra=anti_spectra,
        reference_weights=SNR_sub if (EXOTIC_MOL or len(SUBTRACT) > 0) else None,
    )

    report_result(result, anti_snr=ANTI_SNR)

    output_file_path = save_results(
        result=result,
        file_or_folder=file_or_folder,
        subtract=SUBTRACT,
        species_name=species_name,
        vmr=vmr,
    )
    result["output_file_path"] = output_file_path

    plot_diagnostics(
        result=result,
        show=SHOW,
        err_weight=WEIGHT,
        anti_snr=ANTI_SNR,
        exotic_mol=EXOTIC_MOL,
        species=SPECIES,
        vmr=vmr,
    )

    FancyArg.set_return(output_file_path)
    return result


def report_result(result, anti_snr):
    """Print the same final S/N summary as the original command-line script."""
    ccf_peak = result["ccf_peak"]
    ccf_snr = result["ccf_snr"]
    anti_ccf_snr = result["anti_ccf_snr"]

    if not anti_snr:
        if ccf_snr is not None:
            ccf_peak_str = f"{ccf_peak:.2e}"
            ccf_snr_str = f"{ccf_snr:.2f}"
            Message(f"Peak width: {cstr(peak_kms):c} km/s", "i")
            Message(f"Peak for the CCF: {cstr(ccf_peak_str):g}", "#")
            Message(f"[ RESULT ] SNR for the CCF: {cstr(ccf_snr_str):g}", "#")
        else:
            Message("Could not compute SNR for the CCF, because velocity array is probably too narrow...", "!")
    else:
        ccf_peak_str = f"{ccf_peak:.2e}"
        ccf_snr_str = f"{ccf_snr:.2f}"
        anti_ccf_snr_str = f"{anti_ccf_snr:.2f}"
        Message(f"Peak width: {cstr(peak_kms):c} km/s", "i")
        Message(f"Peak for the CCF: {cstr(ccf_peak_str):g}", "#")
        Message(f"[ RESULT ] SNR for the CCF: {cstr(ccf_snr_str):g}", "#")
        Message(f"SNR for the anti-CCF: {cstr(anti_ccf_snr_str):g}", "#")


def save_results(result, file_or_folder, subtract, species_name, vmr):
    """Save the main CCF and the legacy weight/subtraction outputs."""
    ccf_df = result["ccf_df"]
    spectra = result["spectra"]
    weight_summary = result["weight_summary"]

    output_file_path = file_or_folder.replace('.csv', '.ccf')
    with Task(f"Saving result as: {cstr(output_file_path):b}", new_line=False):
        ccf_df.to_csv(output_file_path, index=False)
        if len(subtract) > 0:
            output_file_path = file_or_folder.replace('.csv', '_subtracted.csv')
            spectra.to_csv(output_file_path, index=False)
        else:
            weight_dir = os.path.dirname(file_or_folder) + "/weights/"
            if not os.path.exists(weight_dir):
                os.mkdir(weight_dir)
            weight_summary.to_csv(
                os.path.dirname(file_or_folder) + f"/weights/w_dict_{species_name}_{vmr}.csv",
                sep=' ',
            )
    return output_file_path


def plot_diagnostics(result, show, err_weight, anti_snr, exotic_mol, species, vmr):
    """Run the original diagnostic plots using returned calculation data."""
    # Local aliases keep the legacy plotting code unchanged where possible.
    SHOW = show
    WEIGHT = err_weight
    ANTI_SNR = anti_snr
    EXOTIC_MOL = exotic_mol
    SPECIES = species
    w_dict = result["w_dict"]
    anti_w_dict = result["anti_w_dict"]
    errors = result["errors"]
    spectra = result["spectra"]
    chunk_labels = result["chunk_labels"]
    spectral_settings = result["spectral_settings"]
    velocities = result["velocities"]
    peak_in = result["peak_in"]
    peak_out = result["peak_out"]
    ccf_df = result["ccf_df"]
    ccf = result.get("ccf_working", result["ccf_df"]["ccf"].to_numpy())
    anti_ccf = result.get("anti_ccf")
    auto_ccf = result.get("auto_ccf")
    ccf_snr_str = result.get("ccf_snr_str")
    anti_ccf_snr_str = result.get("anti_ccf_snr_str")
    ccf_snr = result.get("ccf_snr")
    anti_ccf_snr = result.get("anti_ccf_snr")
    ccf_peak = result.get("ccf_peak")
    noise_sigma = result.get("noise_sigma")
    snr = result.get("snr")
    snr_keys = result.get("snr_keys")
    species_name = result["species_name"]

    ########################
    #### Plot for check ####
    ########################


    if SHOW:
        w = np.array([w_dict[k] for k in w_dict.keys() if 'weight' in k])
        norm = mcolors.Normalize(vmin=min(w), vmax=max(w))
        cmap = plt.cm.viridis
        fig, ax = plt.subplots(figsize=(15, 5))
        with plt.style.context('ggplot'):
            for i,chunk_label in enumerate(chunk_labels):
                chunk = spectra[[f'{chunk_label}.wavelength',f'{chunk_label}.flux', f'{chunk_label}.error.smooth']].dropna()
                try: w_dict[f'{chunk_label}.weight']
                except KeyError: continue
                if(w_dict[f'{chunk_label}.weight']==0): plt.scatter(chunk[f'{chunk_label}.wavelength'], chunk[f'{chunk_label}.flux'], color='k', s=0.5)
                else: plt.scatter(chunk[f'{chunk_label}.wavelength'], chunk[f'{chunk_label}.flux'], color=cmap(norm(w_dict[f'{chunk_label}.weight'])), s=0.5)
                plt.scatter(chunk[f'{chunk_label}.wavelength'], chunk[f'{chunk_label}.error.smooth'], color='darkred', s=0.5)
                if(WEIGHT):
                    try:
                        interp_err = np.interp(chunk[f'{chunk_label}.wavelength'], errors[chunk_label]['wavelength'], errors[chunk_label]['error'])
                        plt.scatter(chunk[f'{chunk_label}.wavelength'], interp_err, color='red', s=0.5, alpha=0.7)
                    except ValueError: continue
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        fig.colorbar(sm, ax=ax, label="Weight")
        plt.ylim(top=0.025, bottom=-0.025)
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('Contrast')
        plt.title(f"Observations")
        plt.show()


    # CCF plot
    mask = np.array([
        np.asarray(w_dict[k]).item() > 3
        for k in w_dict
        if 'snr' in k
    ], dtype=bool)
    try:
        if snr is None: raise NameError("name 'snr' is not defined")
        norm = mcolors.LogNorm(vmin=3, vmax=max(snr[mask]))
    except NameError:
        setting = next(iter(spectral_settings))
        w_dict[f'{setting}.snr'] = float(ccf_snr_str)
        w_dict[f'{setting}.chunk_ccfs'] = ccf
        if(ANTI_SNR):
            anti_w_dict[f'{setting}.snr'] = float(anti_ccf_snr_str)
            anti_w_dict[f'{setting}.chunk_ccfs'] = anti_ccf
        norm = mcolors.Normalize(vmin=np.min(0), vmax=np.max(w_dict[f'{setting}.snr']), clip=True)
    except ValueError: norm = mcolors.LogNorm(vmin=0.1, vmax=max(snr))
    cmap = plt.cm.turbo
    fig, ax = plt.subplots(figsize=(15, 5))
    for spectral_setting in spectral_settings:
        if(w_dict[f'{spectral_setting}.snr']>3):
            if(ANTI_SNR):
                plt.plot(velocities, w_dict[f'{spectral_setting}.chunk_ccfs']/anti_w_dict[f'{spectral_setting}.chunk_ccfs'][~peak_out].std(),
                color=cmap(norm(w_dict[f'{spectral_setting}.snr'])), label=spectral_setting)
            else:
                plt.plot(velocities, w_dict[f'{spectral_setting}.chunk_ccfs']/w_dict[f'{spectral_setting}.chunk_ccfs'][~peak_out].std(),
                color=cmap(norm(w_dict[f'{spectral_setting}.snr'])), label=spectral_setting)
        else:
            plt.plot(velocities, w_dict[f'{spectral_setting}.chunk_ccfs']/w_dict[f'{spectral_setting}.chunk_ccfs'][~peak_out].std(), color='gray')
    if(ANTI_SNR): plt.plot(velocities, ccf_df['ccf']/anti_ccf[~peak_out].std(), color='k', ls='--', label='Full CCF')
    else: plt.plot(velocities, ccf/ccf[~peak_out].std(), color='k', ls='--', label='Full CCF')
    plt.xlabel(r'Velocity [$km\,s^{-1}$]')
    plt.ylabel('S/N')
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, label="SNR")
    plt.legend(fontsize='xx-small')
    plt.grid()
#    plt.savefig(f"/srv/local/anba6163/plots/CCF_{species_name}_{vmr}.png")
    if(SHOW): plt.show()

    ####

    if(ANTI_SNR):
        mask = np.array([
            np.asarray(anti_w_dict[k]).item() > 3
            for k in anti_w_dict
            if 'snr' in k
        ], dtype=bool)
        try:
            if snr is None: raise NameError("name 'snr' is not defined")
            norm = mcolors.LogNorm(vmin=3, vmax=max(snr[mask]))
        except NameError:
            setting = next(iter(spectral_settings))
            norm = mcolors.Normalize(vmin=np.min(0), vmax=np.max(w_dict[f'{setting}.snr']), clip=True)
        except ValueError: norm = mcolors.LogNorm(vmin=0.1, vmax=max(snr))
        cmap = plt.cm.turbo
        fig, ax = plt.subplots(figsize=(15, 5))
        for spectral_setting in spectral_settings:
            if(anti_w_dict[f'{spectral_setting}.snr']>3):
                plt.plot(velocities, anti_w_dict[f'{spectral_setting}.chunk_ccfs']/anti_w_dict[f'{spectral_setting}.chunk_ccfs'][~peak_out].std(),
                    color=cmap(norm(anti_w_dict[f'{spectral_setting}.snr'])), label=spectral_setting)
            else:
                plt.plot(velocities, anti_w_dict[f'{spectral_setting}.chunk_ccfs']/anti_w_dict[f'{spectral_setting}.chunk_ccfs'][~peak_out].std(), 'gray')
        plt.plot(velocities, anti_ccf/anti_ccf[~peak_out].std(), color='k', ls='--', label='Total anti-CCF')
        if(EXOTIC_MOL): plt.plot(velocities, auto_ccf/auto_ccf[~peak_out].std(), color='r', ls='--', label='Autocorrelation function')
        plt.xlabel(r'Velocity [$km\,s^{-1}$]')
        plt.ylabel('S/N')
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        fig.colorbar(sm, ax=ax, label="SNR")
        plt.legend(fontsize='xx-small')
        plt.grid()
 #       plt.savefig(f"/srv/local/anba6163/plots/anti_CCF_{SPECIES}_{vmr}.png")
        if(SHOW): plt.show()

    if(SHOW):
        if(len(spectral_settings)>1):
            # SNR plot
            norm = mcolors.Normalize(vmin=np.min(snr), vmax=np.max(snr), clip=True)
            cmap = plt.cm.GnBu
            fig, ax = plt.subplots(figsize=(10, 10))
            # pie chart
            mask = (snr / np.sum(snr)) < 0.02
            snr_big = snr[~mask]
            snr_small = snr[mask]
            keys_big = np.array(snr_keys)[~mask]
            sorted_ind = np.argsort(snr_big)
            snr = snr_big[sorted_ind]
            snr_keys = keys_big[sorted_ind]
            if len(snr_small) > 0:
                snr = np.append(snr, np.sum(snr_small))
                snr_keys = np.append(snr_keys, 'Other')
            patches, texts, autotexts = plt.pie(snr, colors=cmap(norm(snr)), autopct='%1.1f%%', startangle=-45, pctdistance=0.9,
                                          wedgeprops = {'linewidth': 1, "edgecolor": "black", 'antialiased': True, 'width':0.7}, textprops={'size': 'xx-small'})
            if 'Other' in snr_keys: patches[-1].set_facecolor('gray')
            # arrows
            bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
            kw = dict(arrowprops=dict(arrowstyle="-"), bbox=bbox_props, zorder=0, va="center")
            for i, p in enumerate(patches):
                ang = (p.theta2 - p.theta1)/2. + p.theta1
                y = np.sin(np.deg2rad(ang))
                x = np.cos(np.deg2rad(ang))
                horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
                connectionstyle = f"angle,angleA=0,angleB={ang}"
                kw["arrowprops"].update({"connectionstyle": connectionstyle})
                ax.annotate(snr_keys[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y), horizontalalignment=horizontalalignment, fontsize=6, **kw)
            plt.axis('equal')
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()
