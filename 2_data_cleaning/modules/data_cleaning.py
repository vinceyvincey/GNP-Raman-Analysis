""" Data Cleaning Module """

import pandas as pd
import numpy as np

def clean_outliers(
    df_in: pd.DataFrame,
    group_columns: list,
    fit_features: list,
    extent_multiplier: float = 1.5,
    ):

    """
    Get rid of outliers.
    """
    
    df_out = pd.DataFrame()

    for _, group in df_in.groupby(group_columns):

        for item in fit_features:
            quartile_1 = group[item].quantile(0.25)
            quartile_3 = group[item].quantile(0.75)
            interquartile_range = quartile_3 - quartile_1
            filter1 = (
                group[item] >= quartile_1 - extent_multiplier * interquartile_range) & (
                    group[item] <= quartile_3 + extent_multiplier * interquartile_range)

            group.loc[~filter1, fit_features] = np.nan

        df_out = pd.concat([df_out, group])
        df_out.dropna(subset=fit_features, inplace=True)

    return df_out

def sample_spectra(
    df_in: pd.DataFrame, 
    group_columns: list, 
    spectra_number: int
    ):
    
    """
    Sample a desired number of spectra
    """
    
    df_out = pd.DataFrame()
    for _, group in df_in.groupby(group_columns):
        group_sampled = group.sample(n=spectra_number, random_state=1)
        df_out = pd.concat([df_out, group_sampled])

    df_out = df_out.reset_index(drop=True)

    return df_out