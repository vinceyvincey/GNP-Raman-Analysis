"""Module for Disentangling Contributions from 0D & 1D Defects"""

from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler 

# sns.set_context("paper")
# sns.set_style('whitegrid')
# sns.set(rc={"figure.dpi":300, 'savefig.dpi':300},font_scale = 1.1)


# Fitting Parameters
# =================================================================================================

# Raman Cross Sectional Area Parameters
# -------------------------------------------------------------------------------------------------

raman_csa_regions = ['S','S','A','A']
raman_csa_defect = ['0D','1D','0D','1D']
raman_csa_value = [51,30.3,26.5,30.4]
raman_csa_unit = r'eV$^4$'

raman_csa = pd.DataFrame({
    'region':raman_csa_regions,
    'defect':raman_csa_defect,
    'value':raman_csa_value,
    'unit':raman_csa_unit
})

def filter_raman_csa(raman_csa, region_in:list, defect_in:list):
    filter_df = (raman_csa['region'].isin(region_in)) &\
        (raman_csa['defect'].isin(defect_in))
    out = raman_csa.loc[filter_df,'value'].values[0]
    return out

# Equations from Article
# =================================================================================================

# C.1. Contribution from S-region around point defects.
# -------------------------------------------------------------------------------------------------

def equation_c2(
    L_D,    # Required
    Cs0D    = filter_raman_csa(raman_csa,['S'],['0D'])):
    
    """ Contribution from S-region around point defects. """
    out = Cs0D * equation_c3(L_D)
    return out
    
def equation_c3(
    L_D,    # Required
    r_s     = 2.2):
    
    """ Fraction of the total area occupied by point-defect S-regions. """
    sigma = 1 / (L_D**2)
    out = 1 - np.exp(
        -1 * np.pi * (r_s**2) * sigma
    )
    return out

# C.2. Contribution from S-region near borders.
# -------------------------------------------------------------------------------------------------

def equation_c9(
    L_a,    # Required
    L_D,    # Required
    r_s     = 2.2,
    l_s     = 2,
    Cs1D    = filter_raman_csa(raman_csa,['S'],['1D'])):
    
    """ Contribution from S-region around point defects. """
    sigma = 1 / (L_D**2)
    out = Cs1D * 4 * l_s * (L_a - l_s) * np.exp(-1 * np.pi * (r_s**2) * sigma) / (L_a**2)
    return out

# C.3. Contribution from A-regions (including activation due to both, point defects and borders).
# -------------------------------------------------------------------------------------------------

def equation_c19(
    L_a,    # Required
    L_D,    # Required
    r_s     = 2.2,
    l_s     = 2,
    l_e     = 3.7, 
    Ca1D    = filter_raman_csa(raman_csa,['A'],['1D']),
    Ca0D    = filter_raman_csa(raman_csa,['A'],['0D'])):
    
    """ Contribution from A-regions (including activation due to both, point defects and borders)."""
    sigma = 1 / (L_D**2)
    
    out = 2 * np.pi * Ca0D * l_e * (l_e + r_s) * sigma *\
        (1 - (4 * l_s * (L_a - l_s)/(L_a**2))) *  np.exp(-1 * np.pi * (r_s**2) * sigma) +\
            2 * Ca1D * l_e * (L_a - (2 * l_s))/(L_a**2) *\
                (1-np.exp(-1*((L_a-2*l_s)/l_e))) * np.exp(-1 * np.pi * (r_s**2) * sigma)
    
    return out

# Calculation of (AD/AG)E_l^4.
# -------------------------------------------------------------------------------------------------  

def equation_c1(
    L_a,
    L_D):
    
    """ Calculation of (AD/AG)E_l^4(L_a). """
    ADAG_S_0D = equation_c2(L_D)
    ADAG_S_1D = equation_c9(L_a,L_D)
    ADAG_A_0D1D = equation_c19(L_a,L_D)
    out = ADAG_S_0D + ADAG_S_1D + ADAG_A_0D1D

    return out

# Calculation of (AD/AG)E_l^4(L_a).
# -------------------------------------------------------------------------------------------------  

def equation_2(
    L_a,     # Required
    l_s     = 2,
    l_e     = 4.1, 
    Ca1D    = filter_raman_csa(raman_csa,['A'],['1D']),
    Cs1D    = filter_raman_csa(raman_csa,['S'],['1D'])):
    
    """ Calculation of (AD/AG)E_l^4(L_a). """
    
    out = 1/(L_a**2) * (4 * Cs1D * l_s * (L_a - l_s) +\
        2 * Ca1D * l_e * (L_a - 2 * l_s) *\
            (1-np.exp(-1 * (L_a - 2 * l_s)/l_e)))
    
    return out

# Calculation of (AD/AG)E_l^4(L_D).
# -------------------------------------------------------------------------------------------------  

def equation_3(
    L_D,     # Required
    l_e     = 3.7, 
    r_s     = 2.2,
    Ca0D    = filter_raman_csa(raman_csa,['A'],['0D']),
    Cs0D    = filter_raman_csa(raman_csa,['S'],['0D'])):
    
    """ Calculation of (AD/AG)E_l^4(L_D). """
    sigma = 1 / (L_D**2)
    
    out = Cs0D * (1-np.exp(-1 * np.pi * (r_s**2) * sigma)) +\
        2* np.pi * Ca0D * l_e * (l_e + r_s) * np.exp(-1 * np.pi * (r_s**2) * sigma) * sigma
    
    return out

# Calculation of G band FWHM (linewidth)
# -------------------------------------------------------------------------------------------------  

def equation_1_Ga(
    L_a,
    l_ph        = 16,
    G_fwhm_inf  = 15,
    C_Gamma     = 87):
    
    """Calculation of G band FWHM (linewidth)"""
    xi = L_a
    out = G_fwhm_inf + C_Gamma * np.exp(-1 * xi/l_ph)
    
    return out

def equation_1_Gd(
    L_D,
    alpha       = 10,
    l_ph        = 16,
    G_fwhm_inf  = 15,
    C_Gamma     = 87):
    
    """Calculation of G band FWHM (linewidth)"""
    xi  = alpha * L_D
    out = G_fwhm_inf + C_Gamma * np.exp(-1 * xi/l_ph)
    
    return out

def equation_1_Gd_inv(
    Gd,
    alpha       = 10,
    l_ph        = 16,
    G_fwhm_inf  = 15,
    C_Gamma     = 87):
    
    """Inverse Calculation of G band FWHM (linewidth)"""
    out = -1 * (l_ph/alpha) * np.log((Gd - G_fwhm_inf)/C_Gamma)
    return out

def equation_1_Ga_inv(
    Ga,
    l_ph        = 16,
    G_fwhm_inf  = 15,
    C_Gamma     = 87):
    
    """Calculation of G band FWHM (linewidth)"""
    out = -1 * l_ph * np.log((Ga - G_fwhm_inf)/C_Gamma)
    
    return out

# Simulation Data Scraped from Article
# =================================================================================================

data_dir = Path.cwd().joinpath('modules','simulations')

L_a_sim = pd.read_csv(data_dir.joinpath('Graph_1_La.csv'))
L_D_sim = pd.read_csv(data_dir.joinpath('Graph_2_Ld.csv'))

# Transforming into Density Plot using Machine Learning (Regression)
# ------------------------------------------------------------------------------------------------- 

def data_regression(data,target_col,regressor):
    
    x_train = data.loc[:,['x','y']].values
    y_train = data.loc[:,[target_col]].values
    
    x_train_scaler = StandardScaler().fit(x_train)
    y_train_scaler = StandardScaler().fit(y_train)
    
    x_train_scaled = x_train_scaler.transform(x_train)
    y_train_scaled = y_train_scaler.transform(y_train)
    
    print(f'x_train_scaled: {x_train_scaled.shape}')
    print(f'y_train_scaled: {y_train_scaled.shape}')

    regressor.fit(x_train_scaled, y_train_scaled.ravel())
    
    predictions = regressor.predict(x_train_scaled).reshape(len(x_train_scaled),1)
    print(f'predictions: {predictions.shape}')
    
    data['predictions'] = y_train_scaler.inverse_transform(predictions).ravel()
    
    print(f'Accuracy: {100*regressor.score(x_train_scaled, y_train_scaled):.2f}')
    
    fig,ax = plt.subplots(dpi=100,figsize=(5,7))
    sns.scatterplot(data=data,x=target_col,y='predictions',legend=False)
    ax.plot([0,1],[0,1], transform=ax.transAxes)

    return x_train_scaler,  y_train_scaler, data, regressor
    
def regression_contour(x_train_scaler,y_train_scaler,regressor):
    
    x_axis = np.linspace(10, 85, 1000)
    y_axis = np.linspace(0, 130, 1000)
    
    x_all = np.column_stack((x_axis,y_axis))
    print(f'x_all: {x_all.shape}')
    x_all_scaled = x_train_scaler.transform(x_all)
    
    X, Y = np.meshgrid(x_all_scaled[:,0], x_all_scaled[:,1])
    
    predict_matrix = np.vstack([X.ravel(), Y.ravel()])
    
    prediction = regressor.predict(predict_matrix.T)
    prediction_plot = prediction.reshape(X.shape)
    prediction_plot_actual =  y_train_scaler.inverse_transform(prediction_plot)
    
    return prediction_plot_actual
        
def create_contour(data,regressor,target_col,vmax_in):
    
    x_train_scaler,  y_train_scaler, data, regressor = \
        data_regression(
            data,
            target_col,
            regressor)
    
    prediction_grid = regression_contour(x_train_scaler,y_train_scaler,regressor)
    
    fig,ax  = plt.subplots(figsize=(4,4),dpi=300)
    cp = ax.imshow(prediction_grid,vmin = 0., vmax = vmax_in,
              extent=[10,85, 0, 130])
    plt.colorbar(cp)
    
    outline_data =  create_outline(np.linspace(0.1, 100, 1000))
    sns.lineplot(data=outline_data.loc[outline_data['type'] == 'dashed'],x='Gd',y='y',ls='--',ax=ax,color='k',lw=2)
    sns.lineplot(data=outline_data.loc[outline_data['type'] == 'solid'], x='Ga',y='y',ax=ax,color='k',lw=2)
    
    ax.set_xlim(10,85)
    ax.set_ylim(0,130)


# create_contour(L_a_sim, KNeighborsRegressor(),'La',100)
# create_contour(L_D_sim, KNeighborsRegressor(),'Ld',40)

# Transforming into Density Plot using Interpolation
# ------------------------------------------------------------------------------------------------- 
import matplotlib.tri as tri
from scipy.interpolate import griddata

def interpolate_grid(data,target_col):
    
    data['x'] = data['x'].apply(lambda x: round(x,2))
    data['y'] = data['y'].apply(lambda x: round(x,2))
    triang = tri.Triangulation(data['x'], data['y'])
    interpolator = tri.LinearTriInterpolator(triang, data[target_col])
    
    xi = np.arange(data['x'].min(),data['x'].max(),0.01)
    yi = np.arange(data['y'].min(),data['y'].max(),0.01)
    
    Xi, Yi = np.meshgrid(xi, yi)
    zi = interpolator(Xi, Yi)    
    
    return xi, yi, zi

def interpolate_grid_scipy(data,target_col):
    
    x = data['x'].apply(lambda x: round(x,2))
    y = data['y'].apply(lambda x: round(x,2))
    z = data[target_col]
    
    xi = np.arange(data['x'].min(),data['x'].max(),0.01)
    yi = np.arange(data['y'].min(),data['y'].max(),0.01)
    zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='linear')
    
    return xi, yi, zi

def plot_interpolated_contour(xi, yi, zi,label,data, target_label):
    fig, ax  = plt.subplots(figsize=(4,4),dpi=300)

    ax.contour(xi, yi, zi, levels=14, linewidths=0, colors='k')
    cntr1 = ax.contourf(xi, yi, zi, levels = 12, cmap = 'Reds')

    fig.colorbar(cntr1, ax=ax, label = label)
    
    outline_data =  create_outline(np.linspace(0.1, 100, 1000))
    sns.lineplot(data=outline_data.loc[outline_data['type'] == 'dashed'],x='Gd',y='y',ls='--',ax=ax,color='k',lw=2)
    sns.lineplot(data=outline_data.loc[outline_data['type'] == 'solid'], x='Ga',y='y',ax=ax,color='k',lw=2)
    lines = len(data.loc[:,target_label].unique())
    sns.lineplot(data= data, x='x', y= 'y',hue=target_label,legend=False,lw=0.5,palette = lines*['tab:grey'],alpha=0.25)
    sns.scatterplot(data= data, x='x', y= 'y',color='tab:grey',size=2.5,legend=False,zorder=10,alpha=0.25)
    
    
    ax.set_xlim(10,85)
    ax.set_ylim(0,130)
    ax.set_xlabel(r'$\Gamma _G$ $(cm^{-1})$')
    ax.set_ylabel(r'$(A_D/A_G) \times E_L^4$ $(eV^4)$')
    ax.grid(False)
        
    return fig,ax 

def plot_interpolated_grid(xi, yi, zi, vmax_in):
    fig, ax  = plt.subplots(figsize=(4,4),dpi=300)

    cp = ax.imshow(zi,vmin = 0., vmax = vmax_in,
              extent=[min(xi),max(xi), min(yi), max(yi)])
    plt.colorbar(cp)
    
    outline_data =  create_outline(np.linspace(0.1, 100, 1000))
    sns.lineplot(data=outline_data.loc[outline_data['type'] == 'dashed'],x='Gd',y='y',ls='--',ax=ax,color='k',lw=2)
    sns.lineplot(data=outline_data.loc[outline_data['type'] == 'solid'], x='Ga',y='y',ax=ax,color='k',lw=2)
    
    ax.set_xlim(10,85)
    ax.set_ylim(0,130)
    
    # ax.plot(data['x'], data['y'], 'ko', ms=3) 
    
    return fig,ax 
       
# xi, yi, zi = interpolate_grid_scipy(L_a_sim,'La')
# fig, ax  = plot_interpolated_contour(xi, yi, zi, r'$L_a$ (nm)',L_a_sim, 'La')

# xi2, yi2, zi2 = interpolate_grid_scipy(L_D_sim,'Ld')
# fig, ax  = plot_interpolated_contour(xi2, yi2, zi2, r'$L_D$ (nm)',L_D_sim, 'Ld')

# Plots
# =================================================================================================

def create_outline(range_used = np.linspace(0.1, 100, 1000)):
    
    L_a = np.append(np.ones(len(range_used)) * 500,range_used)
    L_D = np.append(range_used,np.ones(len(range_used)) * 500)
    line_type = ['dashed'] * len(range_used) + ['solid'] * len(range_used) 
    
    data            = pd.DataFrame({'LA':L_a,'LD':L_D})
    data['Gd']      = data['LD'].apply(equation_1_Gd)
    data['Ga']      = data['LA'].apply(equation_1_Ga)
    data['y']       = data.apply(lambda x: equation_c1(x.LA,x.LD), axis=1)
    data['type']    = line_type
    
    return data
    
def plot_outline(range_used = np.linspace(0.1, 100, 1000)):
    
    data = create_outline(range_used)
        
    sns.lineplot(data=data.loc[data['type'] == 'dashed'],x='Gd',y='y',ls='--')
    sns.lineplot(data=data.loc[data['type'] == 'solid'], x='Ga',y='y')
    
    sns.lineplot(data= L_a_sim, x='x', y= 'y',hue = 'La')
    sns.lineplot(data= L_D_sim, x='x', y= 'y',hue = 'Ld')
    
    plt.xlim(10,85)
    plt.ylim(0,130)
    
    plt.show()
    
