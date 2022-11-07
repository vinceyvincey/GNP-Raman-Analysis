"""
Peak Fitting 
"""

import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm
import pybroom as br
from lmfit.models import LorentzianModel, PolynomialModel

def prepare_df(df_in):
    """
    Prepare dataframe for fitting rounds
    """
    df_ready = df_in.unstack().reset_index(drop=True)
    df_ready = pd.DataFrame({"data": df_ready})
    df_ready = df_ready.reset_index(drop=False)
    df_ready.rename(columns={"index": "spectrum"}, inplace=True)

    df_ready["num_params"] = 0
    df_ready["redchi"] = np.nan
    df_ready["pass_test"] = True
    df_ready["fit_round"] = np.nan

    return df_ready


def peak_setup(yaml_file_loc):
    
    with open(yaml_file_loc, 'r') as stream:
        yaml_data = yaml.safe_load(stream)
    
    lmfit_connect = {
        'LorentzianModel':LorentzianModel,
        'PolynomialModel':PolynomialModel}

    background_function =  lmfit_connect[yaml_data['background']['model']['type']](
            yaml_data['background']['model']['order'],
            nan_policy = 'propagate',
            prefix = yaml_data['background']['model']['prefix'])
    
    background_function = {yaml_data['background']['model']['prefix']:background_function}
        
    main_peak = lmfit_connect[yaml_data['main']['model']['type']](
            nan_policy = 'propagate',
            prefix = yaml_data['main']['model']['prefix'])
    
    main_peak = {yaml_data['main']['model']['prefix']:main_peak}

    # additional peaks
    additional_peaks = {}
    for peak in yaml_data['additional_peaks'].items():
            key = peak[1]['model']['prefix']
            value = lmfit_connect[peak[1]['model']['type']](
                    nan_policy = 'propagate',
                    prefix = peak[1]['model']['prefix'])
            
            additional_peaks[key] = value
            
    # peak setpoints
    peak_setpoints = {}
    for peak in yaml_data['additional_peaks'].items():
            
            key = peak[1]['model']['prefix']
            peak_setpoints[key] = {}
            
            for setpoint in peak[1]['setpoints'].items():                
                    value = setpoint[1]
                    setpoint_key  = str(setpoint[0])
                    peak_setpoints[key][setpoint_key]= value
                    
    # add G Peak data
    key = yaml_data['main']['model']['prefix']
    peak_setpoints[key] = {}
    for setpoint in yaml_data['main']['setpoints'].items():                
            value         = setpoint[1]
            setpoint_key  = setpoint[0]
            peak_setpoints[key][setpoint_key]= value
            
    return background_function, main_peak, additional_peaks, peak_setpoints


def fit_spectra(
    x, 
    y, 
    i, 
    errorList, 
    background_function,
    peak_setpoints,
    *args, 
    **kwargs):

    try:

        # (1) Add the Fits together
        # ====================================
        def addFits(fits):
            fit_out = fits[0]
            for fit in fits[1:]:
                fit_out = fit_out + fit
            return fit_out

        mod = addFits(args)

        pars = mod.make_params()

        # (2) Guess Background Centers if Necessary
        # ====================================
        
        if args[-1] == background_function:
            
            # print('Fitting Only Background')
            
            # This only runs if the background is the only item in args
            
            y_background = np.concatenate(
                (y[(x < 1200)], y[(x > 1700) & (x < 2250)], y[(x > 3050)])
            )
            x_background = np.concatenate(
                (x[(x < 1200)], x[(x > 1700) & (x < 2250)], x[(x > 3050)])
            )
            pars = pars.update(background_function.guess(y_background, x=x_background))

        else:
            
            # print('Fitting Multiple Peaks')

            # (3) Take Peak Guesses from Previous Fitting
            # ====================================

            last_fit_params = kwargs["last_fit"]
            variables = last_fit_params.index.values

            for variable in variables:
                pars[variable].set(value=last_fit_params.loc[variable])

            # (4) Add Peak Constraints
            # ====================================
            
            for peak_model in args[1:]:
                
                peak_model_prefix = peak_model.prefix
                
                setpoints = peak_setpoints[peak_model_prefix]
                                
                variable = peak_model_prefix + "center"
                pars[variable].set(
                    min=setpoints["center_min"], max=setpoints["center_max"]
                )
                variable = peak_model_prefix + "amplitude"
                pars[variable].set(min=setpoints["amplitude_min"])

                variable = peak_model_prefix + "sigma"
                pars[variable].set(
                    min=setpoints["sigma_min"], max=setpoints["sigma_max"]
                )
                
            # (5) Guess New Peak Center
            # ====================================
            peak_model = args[-1]
            
            peak_model_prefix = peak_model.prefix
            setpoints = peak_setpoints[peak_model_prefix]
            
            y = np.array(y, dtype="float64")
            x = np.array(x, dtype="float64")
            yvalues = y[(x > setpoints["center_min"]) & (x < setpoints["center_max"])]
            xvalues = x[(x > setpoints["center_min"]) & (x < setpoints["center_max"])]
            
            guesses = peak_model.guess(yvalues, x=xvalues)
            
            for item in ["center", "amplitude", "sigma"]:
                variable = peak_model_prefix + item
                pars[variable].set(value=guesses[variable].value)
                

        # Fit the Curve
        # ====================================

        error_in = errorList.loc[i]
        result = mod.fit(y, pars, x=x, weights=(1 / error_in), max_nfev=1000)

        return result

    except:
        print("dropped")
        return np.nan


def get_fit_data(fit, i, current_round, *args):

    """
    Get fit data
    """

    df_out = pd.DataFrame()

    if str(fit) != "nan":  # if the curve was fit

        # Fit Details - How well the fitting worked
        fit_details = br.glance(fit, var_names="dataset")
        fit_details["spectrum"] = i

        # Fit Parameters
        fit_parameters = (
            br.tidy(fit, var_names="dataset")
            .set_index("name")
            .T.loc[["value"]]
            .reset_index(drop=True)
        )

        fit_parameters = fit_parameters.add_prefix("Par_")

        # Fit Values - Actual Spectra
        fit_values = br.augment(fit, var_names="dataset")
        fit_values_np = fit_values.to_numpy()

        fit_values_df = pd.DataFrame(
            {  #'x':[fit_values_np[:,0]],
                "data": [fit_values_np[:, 1]],
                "best_fit": [fit_values_np[:, 2]],
                "residuals": [fit_values_np[:, 3]],
            }
        )

        # Combine into a single dataframe
        fit_data = pd.concat(
            [fit_details, fit_parameters, fit_values_df], axis=1, join="inner"
        )
        fit_data["fit_round"] = current_round
        if len(args) == 1:
            fit_data["peaks_involved"] = args
        else:
            fit_data["peaks_involved"] = [args]

        df_out = pd.concat([df_out, fit_data])

    return df_out.reset_index(drop=True)


def fits_that_pass(spectrum_fits, this_round):
    filter_1 = spectrum_fits["fit_round"] == this_round
    fits = spectrum_fits.loc[filter_1, :]

    filter_2 = fits["pass_test"] == True
    fits_pass = fits.loc[filter_2, :]

    if fits_pass.shape[0] > 1:
        fits_pass = fits_pass[fits_pass["redchi"] == fits_pass["redchi"].min()]

    return fits_pass


def extract_parameters(previous_fits_pass):

    filter_1 = previous_fits_pass.columns.str.contains(
        "Par_"
    )  #  prefix for parameter columns
    previous_fits_params = previous_fits_pass.loc[:, filter_1].iloc[
        0
    ]  #  find the parameter means
    previous_fits_params.rename(
        lambda x: x[4:], inplace=True
    )  # correct names to match lmfit

    return previous_fits_params


def gather_peaks(previous_fits_pass, args, current_round):

    if current_round == 2:
        old_peaks = previous_fits_pass["peaks_involved"].iloc[0]
        args_in = [old_peaks] + args

    else:
        old_peaks = list(previous_fits_pass["peaks_involved"].iloc[0])
        args_in = old_peaks + args

    return args_in


def check_test_results(previous_fits_pass, fit_results_df, passMark):
    prev_redchi = abs(previous_fits_pass["redchi"] - 1).values
    new_redchi = abs(fit_results_df["redchi"] - 1).values
    difference = new_redchi / prev_redchi
    fit_results_df["pass_test"] = (difference < passMark).astype(bool)
    return fit_results_df


def fit_routine(x, y, i, errorsdf, current_round, background_function,peak_setpoints, args, kwargs={}):

    fit_results = fit_spectra(x, y, i, errorsdf, background_function,peak_setpoints, *args, **kwargs)
    fit_results_df = get_fit_data(fit_results, i, current_round, *args)

    return fit_results_df


def fit_round(df_in, current_round, args, passMark, wn, errorsdf,peak_setpoints):

    df_out = pd.DataFrame()

    for spectrum_number, spectrum_fits in tqdm(df_in.groupby(["spectrum"])):

        fit_results_df = pd.DataFrame()

        # Are there any fits from current round we can build on?
        # --------------------------------------------------------------

        current_fits_pass = fits_that_pass(spectrum_fits, current_round)
        previous_fits_pass = fits_that_pass(spectrum_fits, current_round - 1)

        if current_fits_pass.shape[0] > 0:
            
            # print('Round with different combinations of same number of peaks')

            # Extract the parameters
            # --------------------------------------------------------------

            previous_fits_params = extract_parameters(current_fits_pass)

            # Get previous peaks
            # --------------------------------------------------------------

            args_in = gather_peaks(current_fits_pass, list(args.values()), current_round)

            # Fit data
            # --------------------------------------------------------------

            fit_results_df = fit_routine(
                wn,
                spectrum_fits.iloc[0]["data"],
                spectrum_number,
                errorsdf,
                current_round,
                args_in[0],
                peak_setpoints,
                args_in,
                {"last_fit": previous_fits_params.dropna()},
            )

            # Check test results
            # --------------------------------------------------------------

            if fit_results_df.shape[0] > 0:
                fit_results_df = check_test_results(
                    current_fits_pass, fit_results_df, passMark
                )

        # Are there any fits from previous rounds we can build on?
        # --------------------------------------------------------------

        elif previous_fits_pass.shape[0] > 0:
            
            # print('Fitting Second Round')

            # Extract the parameters
            # --------------------------------------------------------------

            previous_fits_params = extract_parameters(previous_fits_pass)

            # Get previous peaks
            # --------------------------------------------------------------

            args_in = gather_peaks(previous_fits_pass, list(args.values()), current_round) # list

            # Fit data
            # --------------------------------------------------------------

            fit_results_df = fit_routine(
                wn,
                spectrum_fits.iloc[0]["data"],
                spectrum_number,
                errorsdf,
                current_round,
                args_in[0],
                peak_setpoints,
                args_in,
                {"last_fit": previous_fits_params.dropna()},
            )

            # Check test results
            # --------------------------------------------------------------

            if fit_results_df.shape[0] > 0:
                fit_results_df = check_test_results(
                    previous_fits_pass, fit_results_df, passMark
                )

        # Are we on the first fitting round?
        # --------------------------------------------------------------

        elif current_round == 1:
            
            # print('First Round')

            fit_results_df = fit_routine(
                wn,
                spectrum_fits.iloc[0]["data"],
                spectrum_number,
                errorsdf,
                current_round,
                list(args.values())[0],
                peak_setpoints,
                list(args.values()),
            )

            if fit_results_df.shape[0] > 0:
                fit_results_df["pass_test"] = True  # always pass since first round

        df_out = pd.concat([df_out, spectrum_fits, fit_results_df])

    return df_out


def get_final_fits(df_in):

    """
    Keep best fit for each
    """

    df_final_fits = pd.DataFrame()

    for _, spectrum_fits in df_in.groupby(["spectrum"]):

        filter_1 = spectrum_fits["pass_test"] == True
        spectrum_fits_pass = spectrum_fits.loc[filter_1]
        spectrum_keep = spectrum_fits_pass[
            spectrum_fits_pass["redchi"] == spectrum_fits_pass["redchi"].min()
        ]
        spectrum_keep = spectrum_keep.drop(columns=["peaks_involved"], axis=1)
        df_final_fits = pd.concat([df_final_fits, spectrum_keep])

    return df_final_fits