

##############
### Params ###
##############
#%%

DEFAULT_FOLDER = "/home/alst3712/exocomet_project/toy_data/fake_data_1000K_1e+18_0.8cf_47.47SNR.dat" #where full data is?
STD_TRESHOLD = 4
PEAK_WIDTH = 2
CHUNK_GAP_TRESHOLD = 1000 # will be multiplied by the average difference between two consecutive wavelengths
# 1000 means that if the gap between two consecutive wavelengths is more than 1000 times the average gap, we consider it as a new chunk
# 1000 => 1000*0.005nm = 5 nm
CROP_BORDER = 0.8 # nm
STAR_FOLDER = "/home/alst3712/exocomet_project/toy_data" #where cleaned stellar spetrum is?
CUT_TELLURIC = False

###############
### Imports ###
###############

import os
import sys
import numpy as np
# while not os.path.exists('utils'):
#     os.chdir('..')

# print("Current working directory:", os.getcwd())
sys.path.append(os.getcwd())
import pandas as pd
from utils import FancyArg, cstr, Message, FancyIter, Task, suggest_next_command, Baseline

print('started')

# %%

#%%
######################
### Load arguments ###
######################

with FancyArg.create(f"Cut anomalous data and remove telluric lines from a .dat file. Prepare the file for CCF by computing baseline.\
    If {cstr('--f'):c} (file or folder) is a folder, all files in the folder will be processed, by calling recursively this script with --f being a file."):

    FancyArg(
        "f", default=DEFAULT_FOLDER, help="File or folder. If file, this is the complete path to the file that will be preprocessed (must be a .dat file). If folder, all files in the folder will be preprocessed (the complete path to the folder must be given. This folder should contain one sub-folder for each observation, named as '202Y-MM-DD_<band>')."
    )
    FancyArg("std_treshold", default=STD_TRESHOLD, help="The treshold for the std clipping.", type=float)
    FancyArg("peak_width", type=int, default = PEAK_WIDTH, help="In pixels, number of points to remove left an right of each peak.")
    FancyArg("crop_nm", type=float, default = CROP_BORDER, help="Each identified 'chunk' will see it's range of wavelengths cropped by this amount (in nm).")
    FancyArg("cut_telluric", type = bool, default=CUT_TELLURIC, help="Cut the telluric lines from the spectrum ('locate_telluric_lines.py' must have been run before, the script will look for files in <--star_folder>).")
    FancyArg("star_folder", type=str, default=STAR_FOLDER, help="Folder where to search for cleaned stellar spectra. Unused if cut_telluric is false")

# get args and check validity
file_or_folder, STD_TRESHOLD, PEAK_WIDTH, CROP_BORDER, CUT_TELLURIC, STAR_FOLDER = FancyArg.get()
assert os.path.exists(file_or_folder), f"Path {cstr(file_or_folder):r} does not exist. Please check the argument --f."

print(file_or_folder)
if not os.path.isdir(file_or_folder):
    assert file_or_folder.endswith('.dat'), f"File to preprocess must be {cstr('.dat'):r} files, not {cstr(os.path.basename(file_or_folder)):r}"

if CUT_TELLURIC:
    assert os.path.isdir(STAR_FOLDER), f"Folder {cstr(STAR_FOLDER):r} does not exist. Please make sure to provide a valid path to a folder where data has been precomputed by 'cut_tellurics.py'."


#################
### All Files ###
#################

# process all files in one folder
if os.path.isdir(file_or_folder):
    crires_folder = file_or_folder
    # the script will call itself on every file in the folder
    observation_labels = [folder for folder in os.listdir(crires_folder) if os.path.isdir(os.path.join(crires_folder, folder))]
    Message(f"Found {cstr(len(observation_labels)):c} observation folders in the folder {cstr(crires_folder):c}")
    observation_folder_paths = [os.path.join(crires_folder, obs_lbl) for obs_lbl in observation_labels]

    # get every file in all of these folders
    all_files_to_preprocess = np.concatenate([ # contains full path to the file
        [os.path.join(obs_folder_path, file) for file in os.listdir(obs_folder_path) if file.endswith(".dat")]
        for obs_folder_path in observation_folder_paths
    ])

    # sort them by pxl_number / observation label / date
    all_files_to_preprocess = sorted(
        all_files_to_preprocess,
        key = lambda x: (len(os.path.basename(x)), os.path.basename(x), os.path.dirname(x).split("/")[-1].split("_")[1], os.path.dirname(x).split("/")[-1].split("_")[0])
    )

    Message(f"Found {cstr(len(all_files_to_preprocess)):c} files to process.")
    Message.par()

    with Task("Preprocessing all files in the folder", new_line=True):
        for file in FancyIter(all_files_to_preprocess, new_line=True):

            FancyArg.subprocess(
                pythonfile="scripts/preprocess_data.py",
                args={
                    "f": file,
                    "std_treshold": STD_TRESHOLD,
                    "cut_telluric": CUT_TELLURIC,
                    "peak_width": PEAK_WIDTH,
                    "crop_nm": CROP_BORDER,
                    "star_folder": STAR_FOLDER
                },
            )

            Message.par()

    ###############
    ### Goodbye ###
    ###############

    suggest_next_command(
        pythonfile="scripts/merge_observations.py",
        args = {"f": crires_folder, "show": False}
    )
    exit(0)







#################################################################################################################################################################################################################################


# One file!
#######################
### Guess file info ###
#######################
filepath = file_or_folder
filename = os.path.basename(filepath)
output_filename = filename.replace(".dat","_preprocessed.csv")
observation_folder = os.path.dirname(filepath).split("/")[-1]
DATE = observation_folder.split("_")[0]
BAND = observation_folder.split("_")[1]

Message(f"Observation: date='{cstr(DATE):y}', band='{Message.cstr(BAND):y}'", "?")
Message(f"Output file: {cstr(output_filename):c}")


#####################
### Load the file ###
#####################

with Task("Preprocessing the data"):

#changed here
    spectrum = pd.read_csv(
        filepath,
        names = ['wavelength', 'flux', 'error'],
        dtype = np.float64,
        sep = ',',
        header = 0
    )
    Message("Data loaded")

    spectrum_cleaned = spectrum.copy(deep=True)
    spectrum_cleaned = spectrum_cleaned[spectrum_cleaned['flux'].notna()]

    if CUT_TELLURIC:
        with Task("Cutting out the telluric lines"):
            stellar_obs_lbl = f"{DATE}_{BAND}"
            stellar_obs_folder = os.path.join(STAR_FOLDER, stellar_obs_lbl)
            stellar_dat_file = [file for file in os.listdir(stellar_obs_folder) if file.endswith(".dat")]
            assert len(stellar_dat_file) > 0, f"Unable to find a preprocessed stellar file in the folder: {cstr(stellar_obs_folder):r}. The file should be named {cstr('stellar_spectrum_cleaned.dat'):r} and be created by {cstr('locate_telluric_lines.py'):r}"

            star_filepath = os.path.join(stellar_obs_folder, stellar_dat_file[0])

            stellar_spectrum = pd.read_csv(
                star_filepath,
                names = ['wavelength', 'flux'],
                dtype = np.float64,
                sep = ' ',
                header = None
            )
            spectrum_cleaned = spectrum_cleaned[
                spectrum_cleaned['wavelength'].isin(stellar_spectrum['wavelength'])
            ].copy(deep=True)


    with Task("Applying iterative std clipping"):
        #######################
        ### Cut into chunks ###
        #######################
        # define a gap
        gap_treshold = spectrum_cleaned['wavelength'].diff().mean() * CHUNK_GAP_TRESHOLD
        # locate the chunks
        mask = spectrum_cleaned['wavelength'].diff() > gap_treshold
        splits_low = list(
            spectrum_cleaned['wavelength'][mask] # this corresponds to the lower bound of the chunk wavelength range
        )
        splits_high = list(
            spectrum_cleaned['wavelength'][mask.shift(-1, fill_value=False)] # by shifting them all, we get the upper bound of the previous chunk wavelength range
        )

        # locate them in terms of wavelengths
        # define the intervals
        intervals = [(splits_low[i], splits_high[i+1]) for i in range(len(splits_low) - 1)]

        # add the outside borders (minimum and maximum wavelengths)
        intervals = [(spectrum_cleaned['wavelength'].min(), splits_high[0])]\
            + intervals\
                + [(splits_low[-1], spectrum_cleaned['wavelength'].max())]

        # crop each interval
        intervals = [
            (wl_bottom + CROP_BORDER, wl_top - CROP_BORDER) for wl_bottom, wl_top in intervals # very small loss in information (0.3%) with crop
        ]
        
        # actual creation of the chunks
        chunks:list[pd.DataFrame] = []
        for wl_bottom, wl_top in intervals:
            chunks.append(
                spectrum_cleaned[
                    spectrum_cleaned['wavelength'].between(wl_bottom, wl_top)
                ].copy(deep=True)
            )
                

        #############################
        ### Iterative STD removal ###
        #############################


        for i,chunk in enumerate(chunks):
            while True:
                chunk['flux.deviation'] = (chunk['flux'] - chunk['flux'].mean()) / chunk['flux'].std()
                mask = np.abs(chunk['flux.deviation']) > STD_TRESHOLD # what to remove

                # let's broaden the mask with the assumed width of the peaks
                for s in range(PEAK_WIDTH + 1):
                    mask |= mask.shift(s, fill_value=False) \
                        | mask.shift(-s, fill_value=False)
                
                # if nothing to remove, that means the flux is clean!
                if mask.sum() == 0:
                    break
                
                # remove the anomalies
                chunk = chunk[~mask].copy(deep=True)
            chunks[i] = chunk
        # now we merge everyone back together
        spectrum_cleaned = pd.concat(chunks)

               
    # let's take a look at how much data was lost
    data_loss = f"{1 - len(spectrum_cleaned) / len(spectrum):.1%}"
    Message(f"Anomalies removed (data loss: {Message.cstr(data_loss).red()})")


    #######################
    ### Prepare for CCF ###
    #######################

    # as deviation
    spectrum_cleaned['flux.deviation'] = (spectrum_cleaned['flux'] - spectrum_cleaned['flux'].mean()) / spectrum_cleaned['flux'].std()
    # find baseline
    with Task("Computing the baseline..."):
        baseline, params = Baseline().asls(
            spectrum_cleaned['flux'],
            lam = 1e13,
            p = 0.5 # we assume there are as many peaks as valleys # doesn't really matter anyway
        )

        spectrum_cleaned['flux.baseline'] = baseline
        spectrum_cleaned['flux.no_baseline'] = spectrum_cleaned['flux'] - spectrum_cleaned['flux.baseline']
        spectrum_cleaned['flux.no_baseline.deviation'] = (spectrum_cleaned['flux.no_baseline'] - spectrum_cleaned['flux.no_baseline'].mean()) / spectrum_cleaned['flux.no_baseline'].std() # useless too, since cross correlation is scale indepenent


#####################
### Save the file ###
#####################

output_file_path = os.path.join(os.path.dirname(filepath), output_filename)

with Task(f"Saving the preprocessed data to: {cstr(output_file_path):b}", new_line=False):
    spectrum_cleaned.to_csv(output_file_path, index=False)
    
    
        
