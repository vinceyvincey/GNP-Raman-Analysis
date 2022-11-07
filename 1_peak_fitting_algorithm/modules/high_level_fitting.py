"""
Functions to execute program
"""

from pathlib import Path
import pandas as pd
import numpy as np
import logging
from scipy import sparse
from scipy.sparse.linalg import spsolve
from modules.preprocessing.importing import read_wdf
from modules.preprocessing.normalising import truncate_array, truncate_x_array, snv_normalisation
from modules.preprocessing.clean_cosmic_rays import clean_cosmic_rays
from modules.preprocessing.find_error import standard_deviation_regions
from modules.peak_fitting.fit_peaks import prepare_df,fit_round,get_final_fits,peak_setup




def pre_processing(
    file_path,
    min_value=1000,
    max_value=3500
    ):
    """
    Pre processing raw data
    """
    
    # 1.1 Reading Data
    # ------------------------------------------------------------------------------
    
    wdf_data = read_wdf(file_path)
    df_raw = wdf_data["spectra_df"]
    generic_x = wdf_data["x_axis"]
    
    # 1.2 Truncating Data
    # ------------------------------------------------------------------------------
    
    df_trunc = df_raw.applymap(lambda y: truncate_array(generic_x,y,min_value,max_value)[1])
    generic_x = truncate_x_array(generic_x,min_value,max_value)
    
    # 1.3 Normalising Data
    # ------------------------------------------------------------------------------
    
    df_normalised = df_trunc.applymap(snv_normalisation)
    
    # 1.4 Remove Cosmic Rays
    # ------------------------------------------------------------------------------
    
    df_cosmic_removed = df_normalised.applymap(
        lambda y: clean_cosmic_rays(y, m=10, thresh=40)
    )
    
    # 1.5 Find the Error
    # ------------------------------------------------------------------------------
    
    df_errors = (
        df_cosmic_removed.applymap(lambda y: standard_deviation_regions(generic_x, y))
        .unstack()
        .reset_index(drop=True)
    )
    
    # 2.2 Prepare Dataframe for Peak Fitting
    # ------------------------------------------------------------------------------
    
    df_ready = prepare_df(df_cosmic_removed)

    return df_ready, df_errors, generic_x


def pre_processing_cnn(
    file_path,
    min_value=1000,
    max_value=3500
    ):
    """
    Pre processing raw data
    """
    
    # 1.1 Reading Data
    # ------------------------------------------------------------------------------
    
    wdf_data = read_wdf(file_path)
    df_raw = wdf_data["spectra_df"]
    generic_x = wdf_data["x_axis"]
    
    # 1.2 Truncating Data
    # ------------------------------------------------------------------------------
    
    df_trunc = df_raw.applymap(lambda y: truncate_array(generic_x,y,min_value,max_value)[1])
    generic_x = truncate_x_array(generic_x,min_value,max_value)
    
    # 1.3 Normalising Data
    # ------------------------------------------------------------------------------
    
    df_normalised = df_trunc.applymap(snv_normalisation)
    
    
    # 1.4 Remove Cosmic Rays
    # ------------------------------------------------------------------------------
    
    df_cosmic_removed = df_normalised.applymap(
        lambda y: clean_cosmic_rays(y, m=10, thresh=40)
    )
    
    # 1.6 Interpolate Data
    # ------------------------------------------------------------------------------
    
    def interpolate_values(x,y,min_value,max_value):
        
        df = pd.DataFrame({'y':y}, index = x)
        indexList = np.arange(min_value+100, max_value)
        df_interpolated = df.reindex(df.index.union(indexList)).interpolate('index').reset_index()
        df_interpolated.columns = ['x','y']
        df_interpolated = df_interpolated.loc[df_interpolated['x'].isin(indexList)]
        return df_interpolated['y'].values
    
    df_interpolated = df_cosmic_removed.applymap(
        lambda y: interpolate_values(generic_x, y, min_value,max_value)
    )

    # 1.6 Baseline Data
    # ------------------------------------------------------------------------------

    def baseline_als(x, y, lam, p, niter=100):
        L = len(y)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
        w = np.ones(L)
        for i in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)
        return z

    def baseline_spectra(x,y,a=1E7,b=0.001):
        bl = baseline_als(x,y,a, b)
        y2 = y - bl
        return y2

    df_baselined = df_interpolated.applymap(
        lambda y: baseline_spectra(generic_x, y,1E8,0.001)
    )

    # 1.5 Scale Data
    # ------------------------------------------------------------------------------
    
    def scale_values(y):
        y = y + abs(y.min())
        y = y / y.max()
        return y
    
    df_scaled = df_baselined.applymap(
        lambda y: scale_values(y)
    )

    # 2.2 Prepare Dataframe for Peak Fitting
    # ------------------------------------------------------------------------------
    
    df_ready = prepare_df(df_scaled)

    return df_ready, generic_x

def curve_fitting(
    df_ready, 
    df_errors, 
    pass_mark, 
    generic_x,
    yaml_loc,
    ):
    
    """
    Fitting curves in three rounds
    """
    
    # 2.1 Get Peak Constraints from YAML File
    # ------------------------------------------------------------------------------
    
    background_function, main_peak, other_peaks, peak_setpoints = peak_setup(yaml_loc)
    
    # 2.3 Fit Background
    # ------------------------------------------------------------------------------
    
    background_df = fit_round(
        df_ready, 
        1, 
        background_function, 
        pass_mark, 
        generic_x, 
        df_errors,
        peak_setpoints,
        ).reset_index(drop=True)

    logging.info("fitting round 1 complete")
    
    # 2.4 Fit Main Peak
    # ------------------------------------------------------------------------------

    round2_df = fit_round(
        background_df, 
        2,
        main_peak, 
        pass_mark, 
        generic_x, 
        df_errors,
        peak_setpoints,
        ).reset_index(drop=True)

    logging.info("fitting round 2 complete")
    
    # 2.5 Fit Additional Peaks
    # ------------------------------------------------------------------------------

    for i,(key,value) in enumerate(other_peaks.items()):
        
        logging.info(f'fitting {key} peak')

        if i == 0:
            
            round3_df = fit_round(
                round2_df, 
                3, 
                {key:value}, 
                pass_mark, 
                generic_x, 
                df_errors,
                peak_setpoints
                ).reset_index(drop=True)
            
        else:
            round3_df = fit_round(
                round3_df, 
                3, 
                {key:value}, 
                pass_mark, 
                generic_x, 
                df_errors,
                peak_setpoints
                ).reset_index(drop=True)

    return get_final_fits(round3_df)


def export_curve_fittings(df_final_fits, level_1, level_2, level_3, out_name):

    out_location = Path.cwd().joinpath("data", "processed", level_1, level_2, level_3)
    out_location.mkdir(parents=True, exist_ok=True)
    df_final_fits.to_pickle(out_location.joinpath(str(out_name) + ".pkl"))