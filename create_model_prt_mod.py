

###############
### Imports ###
############### 

import os
import sys
import matplotlib
import pandas as pd
while not os.path.exists('utils'):
    os.chdir('..')
# print('Current working directory:', os.getcwd())
sys.path.append(os.getcwd())

from utils import *


INFO_TRESHOLD = 0.1
# BASELINE_KERNEL = 40 # nm # for IPSA # 20 is good to follow a bit the oscillations, 40 may be smoother

# font_size = 25

# # my_rcparams = {
# #     "font.family": "serif",
# #     "font.size": font_size,

# #     "axes.labelsize": font_size,
# #     "axes.titlesize": font_size * 1.2,
# #     "legend.fontsize": font_size * 0.8,
# #     "xtick.labelsize": font_size * 0.9,
# #     "ytick.labelsize": font_size * 0.9,

# #     "xtick.bottom": True,
# #     "xtick.labelbottom": True,
# #     "xtick.top": False,
# #     "xtick.labeltop": False,

# #     "xtick.minor.visible": True,
# #     "ytick.minor.visible": True,

# #     "ytick.right": False,
# #     "ytick.labelright": False,
# #     "ytick.left": True,
# #     "ytick.labelleft": True,

# #     "xtick.direction": "in",
# #     "ytick.direction": "in",

# #     "xtick.major.size": 5,
# #     "xtick.minor.size": 3,
# #     "xtick.major.width": 1,
# #     "xtick.minor.width": 1,
# #     "ytick.major.size": 5,
# #     "ytick.minor.size": 3,
# #     "ytick.major.width": 1,
# #     "ytick.minor.width": 1,

# #     "legend.frameon": False,
# #     "legend.loc": "best",
# #     "legend.shadow": True,
# #     "legend.framealpha": 1,

# #     "axes.linewidth": 1,
# #     "axes.labelcolor": "k",
# #     "axes.edgecolor": "k",
# #     "axes.grid": False,
# #     #"axes.grid.which": "both",
# #     #"axes.xmargin": 0,
# #     #"axes.ymargin": 0,
# #     "xtick.color": "k",
# #     "ytick.color": "k",

# #     "xtick.major.pad": 6,
# #     "ytick.major.pad": 6,
# #     "axes.labelpad": 6,

# #     "hatch.linewidth": 3,
# #     "hatch.color": "w",

# #     "grid.color": "gray",
# #     "grid.alpha": 0.5,

# #     "axes.facecolor": "w",
# # }

# # matplotlib.rcParams.update(my_rcparams)

# ######################
# ### Load Arguments ###
# ######################

# # setup the arguments
# with FancyArg.create(f"This script creates an atmospheric model, thanks to the python package petitRADTRANS."):
    
#     # save path
#     FancyArg(
#         "f", default="data/MODELS", help='Path to the folder where the output will be saved.'
#     )
    
#     # params of the atmosphere
#     FancyArg(
#         "planet", default="bet Pic b", type=str, help="Planet, necessary to compute velocity corrections. Give the EXACT planet name as used by NASA Exoplanet Archive (case and space sensitive). Type 'None' if you don't want to compute velocity corrections"
#     )
#     FancyArg(
#         "wl_range", type=list, default=['1000', '6000'], help='The wavelength range [nm] to consider for the model.'
#     )
    
#     FancyArg(
#         "species", type=list, default=['H2O'], help='The specie(s) used to modell the atmosphere.'
#     )
    
#     FancyArg(
#         "vmr", type=list, default=['1e-4'], help='The volume mixing ratios of the species. Must be in the same order and number as the species.'
#     )
    
#     FancyArg(
#         "T_int", type=float, default=0, help='The temperature [K] to use for the model. If not provided, the temperature used is the one that can be found in the literature.'
#     )
    
#     FancyArg(
#         "radius", type=float, default=0, help='The radius [R_Jup] of the planet to use for the model. If not provided, the radius used is the one that can be found in the literature.'
#     )
    
#     FancyArg(
#         "log_g", type=float, default=0, help='The log(g) [log10 cgs] of the planet to use for the model. If not provided, the log(g) used is the one that can be found in the literature.'
#     )
    
#     FancyArg(
#         "k_IR", type=float, default=0.005, help='The atmopshere IR opacity to use for the model. If not provided, the IR opacity used is the one that can be found in the literature.'
#     )
    
#     # rotationnal broadening
#     FancyArg(
#         "rot", type=float, default=0, help='The rotational broadening [km/s] to apply to the model. If not provided, no rotational broadening is applied.'
#     )
    
#     FancyArg(
#         "epsilon", type=float, default=0.81, help='The value of limb darkening to use for the model. Used only if rotational broadening is applied.'
#     )
    
#     # baseline
#     # FancyArg(
#     #    "baseline", choices=['asls', 'ipsa'], default='ipsa', help='The algorithm to use for the baseline removal. Least squares (ASLS) or Iterative Polynomial Subtraction Algorithm (IPSA).'
#     # )

#     # FancyArg(
#     #     "p", type=float, default=0.01, help='[For ASLS] The p value to use for the baseline removal. The baseline will aim for being at this percentile (between 0 and 1) of the spectrum.'
#     #)

#     # FancyArg(
#     #     "baseline_kernel", type=float, default=DEFAULT_BASELINE_KERNEL, help='[For IPSA] The kernel size (in nm!) to use for the baseline removal. Must be higher than the width of the emission lines, in order not to overfit.'
#     # )

#     # information weight
#     # FancyArg(
#     #     "info_treshold", type=float, default=DEFAULT_INFO_TRESHOLD, help='The treshold to use for the information weight. A region will be considered interesting if above this treshold (as deviation)'
#     # )

#     # signal level # used for error weight
#     FancyArg(
#         "star_file", type=str, default="data/CRIRES/CRIRES_A/bPic1AUSpec_cut.dat", help="Path to the star spectrum file. If invalid path is provided, the result won't contain any 'stellar_flux' column. Star file is assumed to be in log(erg/cm2/s/A) at 1AU."
#     )

#     FancyArg(
#         "show", type=bool, default=True, help="Show intermidiate plots during the process."
#     )

# Message(f"In order to switch from {Message.cstr('HITRAN'):y} to {Message.cstr('POKAZATEL'):y}, modify the file '.petitradtrans' in your home directory!", "?")

# # get the arguments
# FOLDER, PLANET, WL_RANGE, SPECIES, VMR, T_INT, RADIUS, LOG_G, K_IR, ROT, EPSILON, STARFILE, SHOW = FancyArg.get()

# # check the arguments
# assert os.path.isdir(FOLDER), f"{cstr(FOLDER):r} does not exist! Make sure to provide an existing folder."

# if not os.path.isfile(STARFILE):
#     Message(f"Path to the star file {cstr(repr(STARFILE)):r} does not exist! Contrast will not be computed!", "!")
#     COMPUTE_CONTRAST = False
# else:
#     COMPUTE_CONTRAST = True
#     assert STARFILE.endswith('.dat'), f"{cstr(os.path.basename(STARFILE)):r} is not a .dat file! Make sure to provide a .dat file."

# WL_RANGE = [int(wl) for wl in WL_RANGE]

# # set vmrs
# assert not ('H2' in SPECIES or 'He' in SPECIES), f"Species H2 and He are setup directly in the code. Cannot be passed as arguments here. Got {cstr(SPECIES):r}"
# VMR = {SPECIES[i]: float(VMR[i]) for i in range(len(SPECIES))}

# if not ('H2' in VMR):
#     VMR['H2'] = 0.86
# if not ('He' in VMR):
#     VMR['He'] = 0.136

# planet_data = Planet.get(PLANET)

# # set planetary params

# assert RADIUS >=0, f"Radius must be positive, got {cstr(RADIUS):r}"
# assert LOG_G >=0, f"Log(g) must be positive, got {cstr(LOG_G):r}"


# if RADIUS == 0: # not provided, derive from NASA exoplanet archive
#     RADIUS = planet_data.radius / prt_const.r_jup # [cgs] to [R_Jup]

# #    match PLANET:
# #        case 'b':
# #            RADIUS = planet_data.radius / prt_const.r_jup # [cgs] to [R_Jup]
# #        case 'c':
# #            RADIUS = 1.2 # in R_jup

#     radius_as_str = f"{RADIUS:.2e}"
#     Message(f"Using NASA exoplanet archive radius for {PLANET}: {cstr(RADIUS):c} [R_Jup]")
# else:
#     radius_as_str = f"{RADIUS:.2e}"
#     Message(f"Using provided radius: {cstr(radius_as_str):c} [R_Jup]")

# RADIUS *= prt_const.r_jup # [R_Jup] to [cgs]


# if LOG_G == 0: # not provided, use default value
#     g = prt_const.G * planet_data.mass / RADIUS**2 # const is from PRT, so everything is in CGS (except pressure in bar)
#     LOG_G = np.log10(g)

# #    match PLANET:
# #        case 'b':
# #            LOG_G = 4.18
# #        case 'c':
# #            g = prt_const.G * planet_data.mass / RADIUS**2 # const is from PRT, so everything is in CGS (except pressure in bar)
# #            LOG_G = np.log10(g)

#     log_g_as_str = f"{LOG_G:.2f}"
#     Message(f"Using NASA exoplanet archive log(g) for {PLANET}: {cstr(log_g_as_str):c} [log10(cgs)]")
# else:
#     log_g_as_str = f"{LOG_G:.2f}"
#     Message(f"Using provided log(g): {cstr(log_g_as_str):c} [log10(cgs)]")


# if T_INT == 0: # not provided, use default value
#     T_INT = planet_data.equilibrium_temperature
# #    match PLANET:
# #        case 'b':
# #            T_INT = 1700 # K # from Parker et al. 2024
# #        case 'c':
# #            T_INT = 1250 # K from Nowak et al.

#     T_int_as_str = f"{T_INT:.0f}"
#     Message(f"Using NASA exoplanet archive T_int for {PLANET}: {cstr(T_int_as_str):c} [K]")
# else:
#     T_int_as_str = f"{T_INT:.0f}"
#     Message(f"Using provided T_int: {cstr(T_int_as_str):c} [K]")

# #f'prt-{"contrast_" if AS_CONTRAST else "flux_"}wl-{WL_RANGE[0]}nm-{WL_RANGE[1]}nm_species-{"-".join(SPECIES)}_vmr-{"-".join(VMRs)}_{T_int:.0f}K' + (f'_rot-{ROT}-{EPSILON}_{PLANET}' if ROT > 0 else f'_{PLANET}') + '.csv'
# vmrs_as_strs = [f"{v:.2e}" for key, v in VMR.items() if not key in ['H2', 'He']]
# OUTPUT_FILENAME = f'prt-flux_wl-{WL_RANGE[0]:.0f}nm-{WL_RANGE[1]:.0f}nm_species-{"-".join(SPECIES)}_vmr-{"-".join(vmrs_as_strs)}_temp-{T_INT:.0f}K_rad-{RADIUS/prt_const.r_jup:.2f}_logg-{LOG_G:.2f}_kIR-{K_IR:.1e}' + (f'_rot-{ROT:.1f}kms-{EPSILON:.2f}' if ROT > 0 else '') + f'_planet-{PLANET.replace(" ", "")}.csv'
# Message(f"Output filename: {cstr(OUTPUT_FILENAME):c}")
# Message.par()


# ############################
# ### Setting up the model ###
# ############################


# pressure = np.logspace(-6, 2, 100)
# g = 10**LOG_G
# kappa_IR = K_IR

# # let's use the formula from Parker et al. 2024
# temperature = (
#     0.75*T_INT**4 * (2/3 + pressure * u.bar.to(u.g / u.cm / u.s**2) * kappa_IR / g) # convert bars to cgs!
# )**0.25


# MMW = 2.33 * np.ones_like(temperature) # Mean molecular weight, usual for H and He dominated atmospheres


# for key in VMR.keys():
#     VMR[key] *= np.ones_like(temperature)

# mass_fractions = volume_mixing_ratios2mass_fractions(VMR, MMW)

# with Task("Initializing the atmosphere"):
#     atmosphere = Radtrans(
#         pressures = pressure,
#         line_species = SPECIES,
#         wavelength_boundaries = np.array(WL_RANGE) / 1e3, # nm to µm
#         line_opacity_mode = 'lbl',
# #        line_opacity_mode = 'c-k', # for SO2, there's no line-by-line database
#         rayleigh_species=['H2','He'],
#         gas_continuum_contributors = ['H2-H2','H2-He']
#     )


# #####################
# ### Running model ###
# #####################


# with Task("Computing the flux...", new_line = False):
#     wavelengths, flux, _ = atmosphere.calculate_flux(
#         temperatures=temperature,
#         mass_fractions=mass_fractions,
#         reference_gravity=g,
#         mean_molar_masses=MMW
#     )

# wavelengths = wavelengths * u.cm.to('nm') # convert from cm to nm

# Task.par()

# ############## !!!!!!!!!!! ##############
# '''Message('Cose illegali')
# data = pd.read_csv('/srv/local/anba6163/Beta_pic_b_SiO_Models/Bpicb_SiO_pradtrans_output.dat', header=None)

# wavelengths = data.iloc[:,0].to_numpy()*1000
# flux = data.iloc[:,1].to_numpy()

# # to pass from erg cm^-2 s^-1 Hz^-1 to erg cm^-2 s^-1 cm^-1
# c_cm = 2.99792458e10
# wavelengths_cm = wavelengths * u.nm.to('cm')
# nu   = c_cm / wavelengths_cm  # Hz
# flux = flux * nu**2 / c_cm

# cond = (wavelengths < 6000) & (wavelengths > 1000)
# wavelengths = wavelengths[np.where(cond)]
# flux = flux[np.where(cond)]'''


# #############################
# ### Rotational broadening ###
# #############################


# if ROT > 0:
#     with Task("Applying rotational broadening...", new_line = False):
#         flux_norot = flux
#         flux = intRotBroad(
#             wavelengths,
#             flux,
#             EPSILON,
#             ROT
#         )

#     if SHOW:
#         plt.figure(figsize=(10,5))
#         plt.title("Rotational broadening of the spectrum")
#         plt.plot(wavelengths, flux_norot, label='Original', color='darkblue')
#         plt.plot(wavelengths, flux, label=f"Rotationally broadened ({ROT:.0f} km/s)", color='red')
#         plt.xlabel('Wavelength [nm]')
#         plt.ylabel('Flux [cgs]')
#         plt.legend()
#         plt.show()


# spectrum = pd.DataFrame({
#     'wavelength': wavelengths,
#     'flux': flux
# })


# #############################
# ### From flux to contrast ###
# #############################


# if COMPUTE_CONTRAST:
#     with Task("Computing stellar flux...", new_line=True):
#         stellar_spectrum = pd.read_csv(STARFILE, sep=" ", names=['wavelength', 'flux'])

#         # wavelength from Angstrom to nm
#         stellar_spectrum["wavelength"] *= u.Angstrom.to(u.nm)

#         # map in model wavelength
#         stellar_spectrum = pd.DataFrame({
#             "wavelength":spectrum["wavelength"],
#             "flux": np.interp(spectrum["wavelength"], stellar_spectrum["wavelength"], stellar_spectrum["flux"])
#         })

#         # log10(cgs/A) => cgs/A
#         stellar_spectrum["flux"] = 10**stellar_spectrum["flux"]

#         # cgs/A => cgs/cm
#         stellar_spectrum["flux"] /= u.Angstrom.to(u.cm)

#         # (F_planet / F_star)(19.3pc) = F_prt(R_planet) / F_star(1AU) * (R_planet/1AU)**2
#         planet_radius_as_str = f"{RADIUS / prt_const.r_jup:.2f}"
#         Message(f"Radius used for Beta Pictoris {PLANET}: {cstr(planet_radius_as_str):c} R_jup", "i")

#         # STELLAR FLUX IS SCALED HERE SO THAT THE CONTRAST CORRESPONDS TO MODEL['flux'] / STAR['flux'] directly! /!\ Radius dependence is here!
#         stellar_spectrum["flux"] /= (RADIUS * u.cm.to(u.AU))**2 # ouiiiii # RADIUS was in cgs, so we convert it to AU

#         # compute the continuum of the stellar baseline
#         baseline, _ = Baseline().asls(stellar_spectrum["flux"].values, lam=1e13, p=0.95)
#         stellar_spectrum["flux.no_baseline"] = stellar_spectrum["flux"] - baseline # <= 0

#         # pass the info to the model
#         spectrum["stellar_flux"] = stellar_spectrum["flux"]
#         spectrum["stellar_flux.baseline"] = baseline # might be useful at some point
#         spectrum["contrast"] = spectrum['flux'] / stellar_spectrum['flux']


# ########################
# ### Baseline removal ###
# ########################

# with Task(f"Removing the baseline with {cstr('IPSA'):b}...", new_line = False):
#     from pybaselines import Baseline

#     baseline, _ = Baseline().ipsa(
#         spectrum['flux'],
#         half_window = int(BASELINE_KERNEL / spectrum['wavelength'].diff().median()) # 10 nm kernel window
#     )

#     spectrum["flux.no_baseline"] = spectrum["flux"] - baseline

#     if SHOW:
#         fig, axs = plt.subplots(2,1, sharex=True, figsize=(15,8))
#         # plot flux
#         axs[0].plot(spectrum['wavelength'], spectrum["flux"], color='darkblue', label='Flux', zorder = 1)
#         axs[0].plot(spectrum['wavelength'], baseline, color='grey', label='Flux baseline', linestyle='--', zorder = 3)
#         axs[0].plot(spectrum['wavelength'], spectrum["flux.no_baseline"], color='red', label='Flux - baseline', zorder = 2)
#         axs[0].set_ylabel('Flux [cgs/cm]')
#         # plot contrast
#         axs[1].plot(spectrum['wavelength'], spectrum['contrast'], label='Contrast', color='darkblue', zorder = 1)
#         axs[1].plot(spectrum['wavelength'], baseline/stellar_spectrum['flux'], color='grey', label='Contrast baseline', linestyle='--', zorder = 3)
#         axs[1].plot(spectrum['wavelength'], spectrum['contrast'] - baseline/stellar_spectrum['flux'], label='Contrast - baseline', color='red', zorder = 2)
#         axs[1].set_xlabel('Wavelength [nm]')
#         axs[1].set_ylabel('Contrast')
#         plt.title("Baseline removal")
#         plt.legend()
#         plt.show()

#         # plot stellar spectrum
#         plt.figure(figsize=(10,5))
#         plt.title("Stellar flux and model")
#         plt.plot(stellar_spectrum['wavelength'], stellar_spectrum['flux'], label='Scaled stellar flux', color='goldenrod')
#         plt.plot(stellar_spectrum['wavelength'], spectrum["stellar_flux.baseline"], label='Stellar flux baseline', color='gray', linestyle="--", alpha = 0.5)
#         plt.plot(spectrum['wavelength'], spectrum['flux'], label='RadTrans atmospheric planetary model', color='darkblue')
#         plt.xlabel('Wavelength [nm]')
#         plt.ylabel(r'Flux [$\rm erg \dot cm^{-2} \dot s^{-1} \dot cm^{-1}$]')
#         plt.yscale('log')
#         plt.legend()
#         plt.tight_layout()
#         plt.show()


##################################
### Compute information weight ###
##################################


# this has to be done before contrast computation - why???? I changed it

# compute weight as derivative of abs value

spectrum = pd.read_csv('toy_data/model_1000K_1e+18_0.8cf.csv')[['wavelength', 'flux']]
print(spectrum)

step = spectrum["wavelength"].diff().median()
#contrast_no_baseline = spectrum['contrast']-baseline/stellar_spectrum['flux']
spectrum['weight'] = spectrum['flux'].diff().abs().fillna(0) # first weight is NaN because of diff

# smooth out the result
spectrum['weight.smooth'] = spectrum['weight'].rolling(
    int(5 / step), # 5nm window
    center=True
).sum().fillna(0)

spectrum['weight.smooth'] = spectrum['weight.smooth'].rolling(
    int(5 / step),
    center=True
).median().fillna(0)

# rescale for plot and mask
spectrum['weight'] /= spectrum['weight'].std()
spectrum['weight.smooth'] /= spectrum['weight.smooth'].std()
spectrum['weight.mask'] = (spectrum['weight.smooth'] > INFO_TRESHOLD).astype(int)

spectrum.to_csv('co_template_weighted.csv', index=False)
# let's show the result
# if SHOW:
#     fig, axs = plt.subplots(2,1, sharex=True, figsize=(15,8))

#     title = f"Model of {PLANET} with {', '.join(SPECIES)}" + "" if ROT <= 0 else "\n(rotationnaly broadend by [$v_{spin}$"+ f" = {ROT:.1f} km/s, $\epsilon$={EPSILON:.2f}])"
#     #plt.title(title)
#     axs[0].plot(spectrum['wavelength'],spectrum['flux.no_baseline'],color='darkblue',label='Flux Model',alpha=0.5)
#     axs[0].set_ylabel("Flux (cgs cm$^{-1}$)")

#     axs[1].plot(spectrum['wavelength'],spectrum['contrast'] - baseline/stellar_spectrum['flux'],color='darkblue',label='Contrast Model',alpha=0.5)
#     axs[1].set_ylabel("Contrast")
#     axs[1].set_xlabel("Wavelength (nm)")

#     for ax in axs.flat:
#         ax.fill_between(
#             spectrum['wavelength'],
#             spectrum['weight.smooth'],
#             color='goldenrod',
#             alpha=0.5
#         )

#         ax.plot(
#             spectrum['wavelength'],
#             spectrum['weight.smooth'],
#             color='yellow',
#             label='Weight'
#         )

#         ax.axhline(
#             0.1,
#             color='black', linestyle='--', label = 'Threshold'
#         )

#         ybottom, ytop = ax.set_ylim()
#         mask = spectrum['weight.mask'] == 1

#         ax.fill_between(
#             spectrum['wavelength'],
#             ytop,
#             ybottom,
#             where=mask,
#             facecolor='green',
#             zorder=-10,
#             alpha = 0.3,
#             label = "Selected regions"
#         )

#         ax.fill_between(
#             spectrum['wavelength'],
#             ytop,
#             ybottom,
#             where=~mask,
#             facecolor='red',
#             zorder=-10,
#             alpha = 0.3,
#             label = "Discarded regions"
#         )
#         ax.set_ylim(ybottom, ytop)

#     fig.subplots_adjust(wspace=0)
#     handles, labels = axs[0].get_legend_handles_labels()
#     fig.legend(
#         handles,
#         labels,
#         loc="upper right",
#         bbox_to_anchor=(1.01, 0.7),
#         ncol=1,
#         frameon=False,
#     )
#     fig.subplots_adjust(right=0.73)
# #    plt.legend()
#     plt.show()


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
