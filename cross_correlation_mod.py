if __name__ == "__main__":
    import os
    import sys
    while not os.path.exists('utils'):
        os.chdir('..')
    print("Current working directory:", os.getcwd())
    sys.path.append(os.getcwd())


import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u
from astropy import constants as const

#from utils.fancy_print import Message, Task, FancyIterator as FancyIter

c = const.c.to(u.km/u.s).value


def cross_correlate(s1:pd.DataFrame, s2:pd.DataFrame, velocities:np.ndarray, weight:np.ndarray = None, mask:np.ndarray = None, x_axis:list[str] = ["wavelength","wavelength"], y_axis:list[str]=["flux", "flux"], center:bool=False)->np.ndarray:
    """
    Compute cross-crrelation coefficients between two spectras. s1 should be the smallest dataset, since what is computed will be:
    >>> s1.corr(s2.mapped_on(s1))
    This is done this way, since we usually cross correlate spectrum to model, with the model covering a wider range of wavelengths.

    Paramters
    ---------
    s1 (pd.DataFrame): DataFrame representing the first spectrum dataset. Smallest df. Wavelength assumed to be in nm.
    s2 (pd.DataFrame): DataFrame representing the second spectrum dataset. Largest df. Wavelength assumed to be in nm.
    velocities (np.ndarray & units.Quantities): the velocities to compute the cross-correlation on. Assumed to be in km/s.
    weight: (np.ndarray, optional): the weight to apply on the first dataset. Defaults as None.
    mask: (np.ndarray, optional): the mask to apply on the first dataset. Defaults as None (mask will be set to True only values). Allows to remove some part of the data for the computation of the CCF.
    x_axis (list[str],optional): the column names of the x-axis in df1 and df2. This will be the x-axis of the interpolation. Defaults as "wavelength".
    y_axis (list[str],optional): the column names of the y-axis in df1 and df2. This will be the y-axis of the interpolation. Defaults as "flux"

    If x_axis or y_axis are string, they will be converted into a list, eg. x_axis = [x_axis,x_axis]

    Returns
    -------
    np.ndarray: the cross-correlation coefficients. Same shape as velocities
    """

    assert np.all(np.diff(s1[x_axis[0]]) > 0) and\
        np.all(np.diff(s2[x_axis[1]]) > 0), "The x-axis must be monotonic in order to perform cross-correlation"

    if type(x_axis)==str:
        x_axis = [x_axis,x_axis]
    if type(y_axis)==str:
        y_axis = [y_axis,y_axis]

    if mask is None:
        mask = np.ones(len(s1),dtype=bool)

    if weight is None:
        weight = np.ones(len(s1))

    # doppler shift will change the wavelength values, wich might raise an error when mapping the two datasets.
    # Thus, the smallest dataset must be made small enough to fit in the largest dataset, even when shifted by the velocity.


    MARGIN = velocities.max()/c * s2[x_axis[1]].max() # nm # maximal shift due to doppler effect
    MARGIN = np.round(MARGIN) + 1

    if not (
        (s1[x_axis[0]].min() >= s2[x_axis[1]].min() + MARGIN) and (s1[x_axis[0]].max() <= s2[x_axis[1]].max() - MARGIN)
    ):
        print("The x-axis of the first spectrum must be stricly included in the x-axis of the second spectrum (necessary for interpolation) with a margin of 5nm, since velocity shift moves the wavelengths) !"
        "You can try by switching the two spectra, or applying a mask on the first one to reduce wavelength range","!"
        f"Example: df1 = df1[df1.between(df2['wl'].min() + {MARGIN:.0f} ,df2['wl'].max()) - {MARGIN:.0f}]","!"
        "First spectrum is clipped","?")
        # Message("The x-axis of the first spectrum must be stricly included in the x-axis of the second spectrum (necessary for interpolation) (with a margin of 5nm, since velocity shift moves the wavelengths)", "!")
        # Message("You can try by switching the two spectra, or applying a mask on the first one to reduce wavelength range","!")
        # Message(f"Example: df1 = df1[df1.between(df2['wl'].min() + {MARGIN:.0f} ,df2['wl'].max()) - {MARGIN:.0f}]","!")
        # Message("First spectrum is clipped","?")

        clipping_mask = s1[x_axis[0]].between(
            s2[x_axis[1]].min() + MARGIN,
            s2[x_axis[1]].max() - MARGIN
        )
        s1 = s1[
            clipping_mask
        ]
        mask = mask[clipping_mask.values]
        weight = weight[clipping_mask.values]

    # now we can compute the cross correlation
    # for each velocity, we apply velocity shift on one of the two datasets
    # then we map the shifted dataset on the other one
    # finally we compute the correlation coefficient

    out = []

    for v in velocities:

        # doppler shift
        s1_shifted = s1.copy()
        s1_shifted[x_axis[0]] = s1[x_axis[0]] * (1 + v / c)**-1

        # map the second array on the new shifted wavelengths
        # we do this this way because s1 (probably corresponds to the model, covering arbitrary range of wavelengths)
        # is bigger than s2 (the mesured data, covering a smaller range of wavelengths)
        s1_shifted["s2_interpolated"] = np.interp(
            s1_shifted[x_axis[0]],
            s2[x_axis[1]],
            s2[y_axis[1]]
        )
        s1_shifted[y_axis[0]] *= weight
        s1_shifted = s1_shifted[mask] # remove parts we don't want to take into account
        out.append(s1_shifted[y_axis[0]].corr(s1_shifted["s2_interpolated"]))

    out = np.array(out)
    if center:
        out = out - out.mean()

    return out
