"""
Module to support data analysis notebook.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from module_peak_fitting import lorentz


def get_raman_metadata(df_in):
    """
    Extract metadata from Raman file name.
    """
    df_in["material"] = df_in["file"].str.extract(r"^([a-zA-Z0-9]*)_")
    df_in["functionalisation"] = df_in["file"].str.extract(
        r"^[a-zA-Z0-9]*_([a-zA-Z0-9]*)_"
    )
    df_in["plasma_run_date"] = df_in["file"].str.extract(r"_R([a-zA-Z0-9]*)")
    df_in['plasma_run_date'] =  pd.to_datetime(df_in['plasma_run_date'], format = '%y%m%d',errors='ignore')
    df_in['grating'] = df_in["file"].str.extract(r"_G([0-9]*)_")
    df_in["intensity"] = df_in["file"].str.extract(r"_([0-9]*)pc")
    df_in["map_number"] = df_in["file"].str.extract(r"pc_([0-9]*)")

    # df_in.loc[:,df_in.columns.str.contains('Par_')]= \
    #     df_in.loc[:,df_in.columns.str.contains('Par_')].fillna(0)

    df_in["IDG"] = df_in["Par_D_height"] / df_in["Par_G_height"]
    df_in["ADG"] = df_in["Par_D_amplitude"] / df_in["Par_G_amplitude"]
    df_in["FDG"] = df_in["Par_D_fwhm"] / df_in["Par_G_fwhm"]

    df_in["I2DG"] = df_in["Par_D2_height"] / df_in["Par_G_height"]
    df_in["A2DG"] = df_in["Par_D2_amplitude"] / df_in["Par_G_amplitude"]
    df_in["F2DG"] = df_in["Par_D2_fwhm"] / df_in["Par_G_fwhm"]

    return df_in

default_constraints = {
    'IDG':{'min':0,'max':2},
    'I2DG':{'min':0,'max':2},
    'ADG':{'min':0,'max':2},
    'A2DG':{'min':0,'max':2}
}

def filter_constraints(data, constraints_in=default_constraints):
    for item in constraints_in:
        filter_in  = (data[item] < constraints_in[item]['max'])
        data = data.loc[filter_in]
    return data

def compare_stats(data1,data2,groups,target_column,function_in):
    comparison = pd.merge(data1.groupby(groups)[target_column].agg(function_in),
                    data2.groupby(groups)[target_column].agg(function_in),
                    on = groups[-1]).rename(columns={target_column + '_x':'original',target_column + '_y':'constrained'})
    return comparison

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
                group[item] >= quartile_1 - extent_multiplier * interquartile_range
            ) & (group[item] <= quartile_3 + extent_multiplier * interquartile_range)

            group.loc[~filter1, fit_features] = np.nan

        df_out = pd.concat([df_out, group])
        df_out.dropna(subset=fit_features, inplace=True)

    return df_out


def sample_spectra(df_in: pd.DataFrame, group_columns: list, spectra_number: int):
    """
    Sample a desired number of spectra
    """
    df_out = pd.DataFrame()
    for _, group in df_in.groupby(group_columns):
        # spectra_number = spectra_number
        group_sampled = group.sample(n=spectra_number, random_state=1)
        df_out = pd.concat([df_out, group_sampled])

    df_out = df_out.reset_index(drop=True)

    return df_out


col_dict = {
    "ADG": r"A$_D$/A$_G$",
    "IDG": r"I$_D$/I$_G$",
    "FDG": r"$\Gamma_D/\Gamma_G$",
    "I2DG": r"I$_{2D}$/I$_G$",
    "A2DG": r"A$_{2D}$/A$_G$",
    "F2DG": r"$\Gamma_{2D}/\Gamma_G$",
    "Par_G_height": r"I$_G$",
    "Par_G_center": r"$\omega_G$ (cm$^{-1}$)",
    "Par_G_fwhm": r"$\Gamma_G$ (cm$^{-1}$)",
    "Par_D_height": r"I$_D$",
    "Par_D_center": r"$\omega_D$ (cm$^{-1}$)",
    "Par_D_fwhm": r"$\Gamma_D$ (cm$^{-1}$)",
    "Par_D2_height": r"I$_{2D}$",
    "Par_D2_center": r"$\omega_{2D}$ (cm$^{-1}$)",
    "Par_D2_fwhm": r"$\Gamma_{2D}$ (cm$^{-1}$)",
    "O2": r"O$_2$",
    "CF4": r"CF$_4$",
    "NH3": r"NH$_3$",
}

pal = {"O2": "tab:blue", "NH3": "tab:green", "CF4": "tab:purple"}

helpful_params = {"palette": pal, "nice_columns": col_dict}


def kdeplot_functionalisations(df_in, x_values, y_values, material, **kwargs):

    """
    Plot KDE plot comparing functionalisations of different material
    """

    filter1 = (df_in["material"] == material) & (
        df_in["functionalisation"].isin(["O2", "CF4", "NH3"])
    )
    filter2 = (df_in["material"] == material) & (
        df_in["functionalisation"].isin(["BL"])
    )

    pal_in = kwargs["palette"]
    nice_cols = kwargs["nice_columns"]

    graph = sns.FacetGrid(
        data=df_in.loc[filter1],
        col="functionalisation",
        hue="functionalisation",
        height=4,
        aspect=1,
        sharey=True,
        sharex=True,
        palette=pal_in,
        col_order=["O2", "NH3", "CF4"],
    )

    def kdeplot_comparison(*args, **kwargs):
        data = kwargs.pop("data")
        sns.kdeplot(
            data=df_in.loc[filter2],
            x=args[0],
            y=args[1],
            color="k",
            levels=5,
            thresh=0.1,
            fill=False,
        )
        sns.kdeplot(data=data, x=args[0], y=args[1], **kwargs, zorder=10, alpha=0.8)

    graph = graph.map_dataframe(
        kdeplot_comparison, x_values, y_values, fill=True, levels=5, thresh=0.1
    )

    graph.set_titles("{col_name}", weight="bold")
    graph.set_xlabels(nice_cols[x_values])
    graph.set_ylabels(nice_cols[y_values])

    plt.tight_layout()
    # plt.show()

    return graph


def data_for_spectra_plotter(df_in: pd.DataFrame, groups: list, dict_detail: dict):

    # Create DataFrame for Spectral Data
    #  ----------------------------------------------------------------------------------

    df_out = pd.DataFrame()
    for name, group in df_in.groupby(groups[:-1]):

        print(name)

        # Extract Data into Numpy Array
        np_group_x = np.array(group["x"].tolist(), dtype=object)

        # Create ID Column
        group_len = np_group_x.shape[1]
        group_num = np_group_x.shape[0]

        group_id = []
        for i in range(group_num):
            group_id += [i] * group_len

        # Create Location Detail Column

        group_detail = []
        for item in group[groups[-1]]:
            group_detail += [item] * group_len

        # Flatten into DataFrame
        np_group_x = np_group_x.flatten()
        np_group_data = np.array(group["data"].tolist(), dtype=object).flatten()
        np_group_best_fit = np.array(group["best_fit"].tolist(), dtype=object).flatten()

        df_group = pd.DataFrame(
            {
                "data": np_group_data,
                "best_fit": np_group_best_fit,
                "x": np_group_x,
                "id": group_id,
                "detail": group_detail,
            }
        )

        # Add Metadata Back
        df_group[groups[0]] = name[0]
        df_group[groups[1]] = name[1]

        df_out = pd.concat([df_out, df_group])

        df_out.reset_index(drop=True, inplace=True)

    # Prepare DataFrame for Plotting
    #  ----------------------------------------------------------------------------------

    df_out2 = (
        df_out.groupby(groups[:-1] + ["detail", "x"])[["best_fit", "data"]]
        .mean()
        .reset_index(drop=False)
    )

    if len(dict_detail) > 0:
        detail_num = []
        for item in df_out2["detail"].tolist():
            detail_num += [dict_detail[item]]

        df_out2["detail_num"] = detail_num

        for item in ["best_fit", "data"]:
            df_out2[item] = df_out2[item] + df_out2["detail_num"]

    return df_out, df_out2


def component_peak_arrays(df_in, x_array, number_to_sample):

    parameter_columns = df_in.columns[df_in.columns.str.contains("Par_")]
    df_in_parameter_averages = (
        df_in.groupby(["material", "run_date", "functionalisation"])[parameter_columns]
        .median()
        .reset_index(drop=False)
    )

    parameter_columns3 = parameter_columns[parameter_columns.str.contains("Background")]
    df_in_parameter_averages[parameter_columns3] = (
        df_in.groupby(["material", "run_date", "functionalisation"])[parameter_columns3]
        .mean()
        .values
    )

    df_in_parameter_averages_count = (
        df_in.groupby(["material", "run_date", "functionalisation"])[parameter_columns]
        .count()
        .reset_index(drop=False)
    )

    parameter_columns2 = parameter_columns[
        parameter_columns.str.contains("sigma|amplitude")
    ]
    df_in_parameter_averages[parameter_columns2] = df_in_parameter_averages[
        parameter_columns2
    ] * (df_in_parameter_averages_count[parameter_columns2] / number_to_sample)

    peaks_available = parameter_columns.str.extract(r"Par_(\w*)_")[0].unique()

    df_out = pd.DataFrame()
    for i, _ in enumerate(range(df_in_parameter_averages.shape[0])):

        material = df_in_parameter_averages.loc[i, "material"]
        run_date = df_in_parameter_averages.loc[i, "run_date"]
        functionalisation = df_in_parameter_averages.loc[i, "functionalisation"]

        for _, item in enumerate(peaks_available):

            if item != "Background":

                peak_sigma = df_in_parameter_averages.loc[i, "Par_" + item + "_sigma"]
                peak_center = df_in_parameter_averages.loc[i, "Par_" + item + "_center"]
                peak_amplitude = df_in_parameter_averages.loc[
                    i, "Par_" + item + "_amplitude"
                ]

                peak_y_array = lorentz(x_array, peak_sigma, peak_center, peak_amplitude)

            elif item == "Background":
                peak_amplitude = 40
                ppar = []
                for j in range(4):
                    ppar = np.append(
                        ppar,
                        df_in_parameter_averages.loc[i, "Par_" + item + "_c" + str(j)],
                    )
                peak_y_array = np.polyval(np.flip(ppar), x_array)

            if peak_amplitude > 10:

                df_peak_arrays = pd.DataFrame(
                    {
                        "x": x_array,
                        "y": peak_y_array,
                        "peak": item,
                        "material": material,
                        "run_date": run_date,
                        "functionalisation": functionalisation,
                    }
                )

                df_out = pd.concat([df_peak_arrays, df_out])

    df_out.reset_index(drop=True, inplace=True)
    return df_out, df_in_parameter_averages


def get_all_components(df_sampled, x_array, number_to_sample):

    df_spectra_components, _ = component_peak_arrays(
        df_sampled, x_array, number_to_sample
    )
    _, df_out = data_for_spectra_plotter(
        df_sampled, ["material", "run_date", "functionalisation"], {}
    )
    df_out.rename(columns={"detail": "functionalisation"}, inplace=True)
    df_main_components = df_out.melt(
        id_vars=["material", "run_date", "functionalisation", "x"],
        value_vars=["data", "best_fit"],
    ).rename(columns={"variable": "peak", "value": "y"})
    df_all_components = pd.concat(
        [df_main_components, df_spectra_components]
    ).reset_index(drop=True)

    df_all_components_bg = pd.DataFrame()
    df_all_components.sort_values(
        ["material", "run_date", "functionalisation", "x"], inplace=True
    )
    for _, group in df_all_components.groupby(
        ["material", "run_date", "functionalisation"]
    ):

        filter1 = group["peak"] == "Background"
        filter2 = group["peak"] == "data"
        filter3 = group["peak"] == "best_fit"
        group.loc[filter2, "y"] = (
            group.loc[filter2, "y"].values - group.loc[filter1, "y"].values
        )
        group.loc[filter3, "y"] = (
            group.loc[filter3, "y"].values - group.loc[filter1, "y"].values
        )
        group.loc[filter1] = np.nan
        df_all_components_bg = pd.concat([df_all_components_bg, group])

    return df_all_components_bg, df_all_components


def average_raman_plot(df_in, flag=0):

    graph = sns.FacetGrid(
        data=df_in,
        col="functionalisation",
        row="material",
        hue="peak",
        height=4,
        aspect=1,
        sharey="row",
        sharex=True,
        palette="tab10",
        col_order=["BL", "O2", "NH3", "CF4"],
        legend_out=True,
    )

    def deconvoluted_spectrum_plot(*args, **kwargs):
        data = kwargs.pop("data")
        peaks_available = data["peak"].unique()
        for item in peaks_available:
            if flag == 1:
                if item not in ["data", "best_fit", "Background"]:
                    filter1 = data["peak"] == item
                    x_in = data.loc[filter1, args[0]]
                    y_in = data.loc[filter1, args[1]]
                    plt.fill_between(x_in, y_in, alpha=0.75, lw=0, **kwargs)
            if item in ["data"]:
                filter1 = data["peak"] == item
                x_in = data.loc[filter1, args[0]]
                y_in = data.loc[filter1, args[1]]
                plt.scatter(x_in, y_in, s=10, color="tab:grey")
            elif item in ["best_fit"]:
                filter1 = data["peak"] == item
                x_in = data.loc[filter1, args[0]]
                y_in = data.loc[filter1, args[1]]
                plt.plot(x_in, y_in, color="k")

    graph = graph.map_dataframe(deconvoluted_spectrum_plot, "x", "y")

    graph.set_titles("{col_name}", weight="bold")
    graph.set_xlabels("Raman shift (cm$^{-1}$)")
    graph.set_ylabels("Intensity (a.u.)")
    graph.set(yticklabels=[])

    return graph
