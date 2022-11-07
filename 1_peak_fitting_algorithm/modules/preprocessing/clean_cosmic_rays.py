'''
Clean Cosmic Rays
'''

import numpy as np
import pandas as pd

def modified_z_score(intensity):
    median_int = np.median(intensity)
    mad_int = np.median([np.abs(intensity - median_int)])
    modified_z_scores = 0.6745 * (intensity - median_int) / mad_int
    return modified_z_scores


def fixer(y, m, thresh):
    threshold = thresh  # binarization threshold.
    spikes = abs(np.array(modified_z_score(np.diff(y)))) > threshold
    y_out = y.copy()  # So we don’t overwrite y
    for i in np.arange(len(spikes)):
        if (i > m) & (i < len(y) - m):
            if spikes[i] != 0:  # If we have an spike in position i
                y_out[i - m : i + m] = np.nan  # and we average their values
        return y_out


def interpolate_series(y):
    sy = pd.Series(y)
    sy = sy.interpolate(method="linear")
    return sy


def clean_cosmic_rays(y, m, thresh):
    y = fixer(y, m, thresh)
    sy = interpolate_series(y)
    return sy.values

