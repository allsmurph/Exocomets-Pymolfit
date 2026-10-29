
###############
### Imports ###
###############

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
while not os.path.exists("utils"):
    os.chdir("..")
# print("Current working directory:", os.getcwd())
sys.path.append(os.getcwd())

from utils import FancyArg, cstr, Message, FancyIter, Task, suggest_next_command


######################
### Load arguments ###
######################

# two argument, the folder to search, that contains n subfolders, and the filename to search in each subfolder
with FancyArg.create("Merge all the files with a certain name in a folder"):

    FancyArg(
        "f", default="/home/alst3712/exocomet_project/clean_data", help="File or folder. You need to prvide the path to the file or folder, not only the name. If file, all files with the same name in different observation folders will by merged together. Must be a .csv file. If folder, it must contain different subfolders corresponding to different observations. The script will then call itself on all the different file that it is able to find."
    )
    # FancyArg(
    #     "planet", default="b", choices=['b', 'c'], help="Planet, necessary to compute velocity corrections."
    # )
    FancyArg(
        "chunk_gap", type=float, default=1000, help="Chunk gap treshold. Used to split the spectra into chunks. The lower the value, the more chunks will be created."
    )
    FancyArg(
        "show", type=bool, default=True, help="Show the plots during the process."
    )

# get arg and check validity
file_or_folder, CHUNK_GAP_TRESHOLD, SHOW = FancyArg.get()
assert os.path.exists(file_or_folder), f"Path {cstr(file_or_folder):r} does not exist!"

if not os.path.isdir(file_or_folder):
    assert file_or_folder.endswith('.csv'), f"File must be a {cstr('.csv'):r} file, not {cstr(os.path.basename(file_or_folder)):r}!"



#################
### All files ###
#################

if os.path.isdir(file_or_folder):
    crires_folder_path = file_or_folder

    # find one observation folder to look at
    observation_folders = [folder for folder in os.listdir(crires_folder_path) if os.path.isdir(os.path.join(crires_folder_path, folder)) and folder!='weights']
    assert len(observation_folders) > 0, f"No folder found in {cstr(crires_folder_path):r}. Please make sure to provide the data."

    first_obs_folder = observation_folders[0]
    first_order_folder = os.path.join(
        crires_folder_path, first_obs_folder, "order_regions"
    )

    assert os.path.isdir(first_order_folder), (
        f"Could not find order_regions folder in {first_obs_folder}"
    )

    distinct_filenames = {
        filename
        for filename in os.listdir(first_order_folder)
        if filename.endswith("_preprocessed.csv")
    }

    distinct_filenames = sorted(distinct_filenames)

    distinct_filenames = sorted(
        distinct_filenames,
        key = lambda x: (len(x), x)
    )

    output_files = []
    for file in FancyIter(distinct_filenames, new_line=True):
        #first_observation_folder = [folder for folder in os.listdir(crires_folder_path) if os.path.isdir(os.path.join(crires_folder_path, folder)) and folder!='weights'][0]
        with Task(f"Merging files: {cstr(file):b}"):
            FancyArg.subprocess(
                pythonfile="merge_observations_mod.py",
                args = {
                    "f": os.path.join(
                        crires_folder_path, first_obs_folder, "order_regions", file
                    ),
                    #"planet": planet,
                    "chunk_gap": CHUNK_GAP_TRESHOLD,
                    "show": SHOW
                }
            )
            output_files.append(FancyArg.pop_return())
        Message.par()
    Message(f"All files merged for folder {cstr(crires_folder_path):g}. Ready for cross correlation. Use {cstr('create_model_prt.py'):g} to generate the model. Use {cstr('weighted_ccf.py'):g} to compute the cross correlation.", "#")

    # suggest_next_command(
    #     "scripts/weighted_ccf.py",
    #     args= {
    #         "f": crires_folder_path,
    #         "model": "data/MODELS/prt-flux_wl-2800nm-4800nm_species-H2O_vmr-1.00e-04_temp-1700K_rad-1.65_logg-4.18_kIR-5.0e-03_planet-b.csv",
    #     }
    # )
    print(
    f"Next step: python scripts/weighted_ccf.py "
    )
    FancyArg.set_return(output_files)
    exit(0)










#################################################################################################################################################################################################################################

# filename != ""
#######################################
### Locate all files with same name ###
#######################################

file_path = Path(file_or_folder).resolve()
filename = file_path.name

# Expected structure:
# clean_data/observation_name/order_regions/filename.csv
crires_folder_path = str(file_path.parent.parent.parent)

# Identify the observation corresponding to the supplied file
first_obs_folder = file_path.parent.parent.name

Message(f"Identified main folder as: {cstr(crires_folder_path):y}", "?")

observation_folders = sorted([
    folder for folder in os.listdir(crires_folder_path)
    if os.path.isdir(os.path.join(crires_folder_path, folder))
    and folder != "weights"
    and os.path.isdir(
        os.path.join(crires_folder_path, folder, "order_regions")
    )
])

assert filename.endswith(".csv"), "Input file must be a .csv file."
assert first_obs_folder in observation_folders, (
    f"Could not identify observation folder: {first_obs_folder}"
)

# Replace the observation name in the filename to create a template.
# Example:
# tellurics_corrected_version_0.8.3_2025-09-27_M4318_base_preprocessed.csv
# becomes:
# tellurics_corrected_version_0.8.3_{OBS}_base_preprocessed.csv
if first_obs_folder not in filename:
    raise ValueError(
        f"The filename {filename!r} does not contain the observation "
        f"name {first_obs_folder!r}. Cannot automatically match files "
        "between observations."
    )

filename_template = filename.replace(first_obs_folder, "{OBS}")

file_paths_to_merge = {}

for obs_folder in observation_folders:
    matching_filename = filename_template.replace("{OBS}", obs_folder)

    filepath = os.path.join(
        crires_folder_path,
        obs_folder,
        "order_regions",
        matching_filename,
    )

    if os.path.isfile(filepath):
        file_paths_to_merge[obs_folder] = filepath
    else:
        print(f"Skipping {obs_folder}: file not found: {matching_filename}")

assert len(file_paths_to_merge) > 0, (
    f"No matching files found for template: {filename_template}"
)

Message(f"Found {len(file_paths_to_merge)} files to merge.")

##################
### Load files ###
##################

with Task("Loading the files", new_line=False):
    spectra_to_merge = {
        observation_label: pd.read_csv(filepath) for observation_label, filepath in file_paths_to_merge.items()
    }

print(spectra_to_merge)
###########################
### Velocity Correction ###
###########################

# c = astro_const.c.to('km/s').value
# with Task(f"Collecting radial velocites for planet {cstr(planet):b}"):
#     rv_corrections = []
#     for observation_label, spectrum in spectra_to_merge.items():
#         date = observation_label.split("_")[0]

#         rv_tot = Velocity.get(planet, date)['vrad_tot']

#         rv_corrections.append(rv_tot) # no need to keep this, but I want to print its average at some point

#         # doppler redshift
#         spectrum['wavelength'] = spectrum['wavelength'] * (1 + rv_tot / c) ** -1 # inplace modiication

#     Message(f"Average planet radial velocity: {Message.cstr(f'{np.mean(rv_corrections):.1f} km/s').red()}")


#######################
### Cut into Chunks ###
#######################

# since we want to compute one ccf for each chunk,
# who will be weighted depending on the error and
# the quantitiy of information in each model,
# we need to cut the spectra into chunks, again

# wl_min = min([spectrum['wavelength'].min() for spectrum in spectra_to_merge.values()])
# wl_max = max([spectrum['wavelength'].max() for spectrum in spectra_to_merge.values()])
# Message(f"Wavelength range: {wl_min:.1f} to {wl_max:.1f} nm")
# all_chunks = []

# for observation_label, spectrum in FancyIter(spectra_to_merge.items()):

#     # split into chunks
#     chunk_treshold = spectrum['wavelength'].diff().median() * CHUNK_GAP_TRESHOLD
#     mask = spectrum['wavelength'].diff() > chunk_treshold

#     splits = [spectrum['wavelength'].min()]\
#         + list(spectrum['wavelength'][mask])\
#             + [spectrum['wavelength'].max() + 1]

#     intervals = [(splits[i], splits[i+1]) for i in range(len(splits) - 1)]
#     chunks = [
#         spectrum[
#             spectrum['wavelength'].between(*interval, inclusive='left')
#         ] for interval in intervals
#     ]

#     # append each chunk
#     for i, chunk in enumerate(chunks, start=1):
#         chunk_on_basis = pd.DataFrame()
#         chunk_column_name = f'{observation_label}_{i}'
#         chunk_on_basis[f'{chunk_column_name}.wavelength'] = chunk['wavelength'].reset_index(drop=True)
#         for column_suffix in ['flux', 'flux.no_baseline', 'error', 'additional_error']:
#             try: chunk_on_basis[f'{chunk_column_name}.{column_suffix}'] = (chunk[column_suffix].reset_index(drop=True))
#             except KeyError: print(f"No {column_suffix} for {chunk_column_name}")
#         # add the smooth error column
#         N_ROLLING = 1000
#         chunk_on_basis[f'{chunk_column_name}.error.smooth'] = chunk_on_basis[f'{chunk_column_name}.error'].rolling(N_ROLLING, center=True, min_periods=N_ROLLING//2).median()
# #        chunk_on_basis[f'{chunk_column_name}.additional_error.smooth'] = chunk_on_basis[f'{chunk_column_name}.additional_error'].rolling(N_ROLLING, center=True,min_periods=N_ROLLING//2).median()
#         all_chunks.append(chunk_on_basis)

# Message(f"Split all spectra into {cstr(len(all_chunks)):c} chunks.")


# #########################
# ### Perform the merge ###
# #########################


# spectra = pd.concat(
#     [chunk for chunk in all_chunks],
#     axis = 1
# )
# #print(spectra)
# #for element in np.where(spectra!=None):
# #        print(len(element))
# #        print(element)
# ##############################
# ### Show some random chunk ###
# ##############################

# if SHOW:
#     observation_labels = [col.replace('.flux','')[:-2] for col in spectra.columns if col.endswith('.flux')]
#     observation_labels = list(set(observation_labels))
#     observation_label = np.random.choice(observation_labels)

#     # let's plot flux.no_baseline, error and smooth error
#     chunk_labels = list(set([col.split('.')[0] for col in spectra.columns if observation_label in col]))

#     with plt.style.context('ggplot'):
#         plt.figure(figsize=(15, 5))
#         for i,chunk_label in enumerate(chunk_labels):
#             if i == 0:
#                 plt.scatter(spectra[chunk_label + '.wavelength'], spectra[chunk_label + '.flux.no_baseline'], color='grey', label='flux.no_baseline', s=0.5)
#                 plt.plot(spectra[chunk_label + '.wavelength'], spectra[chunk_label + '.error'], color='red', linewidth=0.25, label='error')
#                 plt.plot(spectra[chunk_label + '.wavelength'], spectra[chunk_label + '.error.smooth'], color='darkred', linewidth=1, label='error.smooth')
# #                plt.plot(spectra[chunk_label + '.wavelength'], spectra[chunk_label + '.additional_error'], color='orange', linewidth=1, label='additional_error')
#             plt.scatter(spectra[chunk_label + '.wavelength'], spectra[chunk_label + '.flux.no_baseline'], color='grey', s=0.5)
#             plt.plot(spectra[chunk_label + '.wavelength'], spectra[chunk_label + '.error'], color='red', linewidth=0.25)
#             plt.plot(spectra[chunk_label + '.wavelength'], spectra[chunk_label + '.error.smooth'], color='darkred', linewidth=1)
# #            plt.plot(spectra[chunk_label + '.wavelength'], spectra[chunk_label + '.additional_error'], color='orange', linewidth=1)
#         plt.legend()
#         plt.xlabel('Wavelength [nm]')
#         plt.ylabel('Contrast')
#         plt.title(f"Observation: {observation_label}")
#         plt.savefig('/plots/merged_spectra.png')

#     plt.show()



# ######################
# ### Save into 1 df ###
# ######################

# output_file_path = os.path.join(crires_folder_path, filename.replace('.csv', '_allspectra.csv'))

# with Task(f"Saving merged observations to: {cstr(output_file_path):g}", new_line=False):

#     spectra.to_csv(
#         output_file_path,
#         index = False
#     )

# FancyArg.set_return(output_file_path)


################################
### Process each observation ###
################################

output_files = []

for observation_label, spectrum in FancyIter(spectra_to_merge.items()):

    Message(
        f"Processing {cstr(observation_label):b}: "
        f"{spectrum['wavelength'].min():.1f}–"
        f"{spectrum['wavelength'].max():.1f} nm"
    )

    # Sort by wavelength before identifying gaps
    spectrum = spectrum.sort_values("wavelength").reset_index(drop=True)

    # Split into chunks at large wavelength gaps
    chunk_threshold = (
        spectrum["wavelength"].diff().median() * CHUNK_GAP_TRESHOLD
    )
    mask = spectrum["wavelength"].diff() > chunk_threshold

    splits = (
        [spectrum["wavelength"].min()]
        + list(spectrum.loc[mask, "wavelength"])
        + [spectrum["wavelength"].max() + 1]
    )

    intervals = [
        (splits[i], splits[i + 1])
        for i in range(len(splits) - 1)
    ]

    chunks = [
        spectrum[
            spectrum["wavelength"].between(
                *interval, inclusive="left"
            )
        ]
        for interval in intervals
    ]

    # Create the same wide, chunk-based structure as before
    all_chunks = []

    for i, chunk in enumerate(chunks, start=1):
        if chunk.empty:
            continue

        chunk_on_basis = pd.DataFrame()
        chunk_column_name = f"{observation_label}_{i}"

        chunk_on_basis[f"{chunk_column_name}.wavelength"] = (
            chunk["wavelength"].reset_index(drop=True)
        )

        for column_suffix in [
            "flux", "flux.no_baseline", "error", "additional_error"
        ]:
            if column_suffix in chunk.columns:
                chunk_on_basis[f"{chunk_column_name}.{column_suffix}"] = (
                    chunk[column_suffix].reset_index(drop=True)
                )

        if f"{chunk_column_name}.error" in chunk_on_basis.columns:
            N_ROLLING = 1000
            chunk_on_basis[f"{chunk_column_name}.error.smooth"] = (
                chunk_on_basis[f"{chunk_column_name}.error"]
                .rolling(
                    N_ROLLING,
                    center=True,
                    min_periods=N_ROLLING // 2
                )
                .median()
            )

        all_chunks.append(chunk_on_basis)

    if not all_chunks:
        print(f"No chunks found for {observation_label}; skipping.")
        continue

    # Merge this observation's chunks side by side
    merged_spectrum = pd.concat(all_chunks, axis=1)

    # Save alongside the input CSV
    input_path = Path(file_paths_to_merge[observation_label])
    output_path = input_path.with_name(
        input_path.stem + "_merged.csv"
    )

    merged_spectrum.to_csv(output_path, index=False)
    output_files.append(str(output_path))

    Message(f"Saved: {cstr(output_path):g}")

FancyArg.set_return(output_files)
