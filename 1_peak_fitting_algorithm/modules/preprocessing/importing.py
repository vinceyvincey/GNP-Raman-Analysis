"""
Importing Files
"""

import pandas as pd
import numpy as np
from renishawWiRE import WDFReader

def read_wdf(filename):
    """
    Extract Data from .wdf map
    """
    reader = WDFReader(filename)
    spectra = reader.spectra

    if reader.measurement_type == 3:  # map
        w, h = reader.map_shape
        if (w > 1) & (h > 1):
            df = pd.DataFrame(zip(*spectra))
        else:
            x_length = len(reader.xdata)
            spectra = spectra.reshape(w, h, x_length)
            df = pd.DataFrame(zip(*spectra))

    elif reader.measurement_type == 1:  # point
        w = np.nan
        h = np.nan
        df = pd.DataFrame([[spectra]])
    else:  # unspecified
        w = np.nan
        h = np.nan
        df = pd.DataFrame([[spectra]])

    wdf_data = {
        "x_axis": reader.xdata,
        "spectra_list": spectra,
        "spectra_df": df,
        "x_position": reader.xpos,
        "y_position": reader.ypos,
        "width": w,
        "height": h,
        "laser": reader.laser_length,
        "measurement_type": reader.measurement_type,
        "accumulations": reader.accumulation_count,
        "size": reader.count,
    }

    return wdf_data
