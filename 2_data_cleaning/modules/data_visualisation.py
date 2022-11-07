""" Data Visualisation Module """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def lorentz_data(data_in:pd.DataFrame,peak:str):
    """ Get Lorentz Data. """
  
    sigma = data_in["Par_" + peak + "_sigma"]
    center = data_in["Par_" + peak + "_center"]
    amplitude = data_in["Par_" + peak + "_amplitude"]
    
    return sigma, center, amplitude

def background_polynomial_data(data_in:pd.DataFrame,poly_order:int,x_array:np.ndarray):
    """ Background polynomial data. """

    ppar = []
    for j in range(poly_order+1):
        bg_value = data_in["Par_Background_c" + str(j)]
        ppar = np.append(ppar,bg_value)
        
    peak_y_array = np.polyval(np.flip(ppar), x_array)
    return peak_y_array

def lorentz(x, sigma, center, A):
    """ Lorentzian function as in LMFIT module. """

    LHS = A / np.pi
    RHS = sigma / ((x - center) ** 2 + sigma ** 2)
    out = LHS * RHS
    return out

def plot_single_fit(
    data,
    plot_number,
    x,
    background_poly = 3,
):
    
    """
    Plot a single fit
    """
    
    # fitting = data.loc[data['spectrum']==plot_number]
    fitting = data.loc[plot_number]
    
    y_bg = background_polynomial_data(fitting,background_poly,x) # Get the background 
    
    y = fitting['data'] # Raw Data
    bf = fitting['best_fit'] # Best Fit
    
    y_c  = y - y_bg # Baseline Corrected Raw Data
    bf_c  = bf - y_bg # Baseline Corrected Best Fit
    
    fig,ax = plt.subplots(figsize=(5*(3/4),5))
    plt.scatter(x,y_c,s=2,c='tab:grey',alpha=0.5,label='raw')
    plt.plot(x,bf_c,c='k',label='fit')
    
    for peak in fitting['peaks'][1:]:
        sigma, center, amplitude = lorentz_data(fitting, peak)
        y_fit = lorentz(x,sigma,center,amplitude)
        plt.plot(x,y_fit,lw=1,label=peak)
        
    
    plt.legend()
    
    return fig,ax

