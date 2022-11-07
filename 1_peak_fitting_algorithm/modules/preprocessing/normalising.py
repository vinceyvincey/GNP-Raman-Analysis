'''
Normalising Data
'''

from pyspectra.transformers.spectral_correction import snv

def truncate_array(array_x,array_y,min_val,max_val):
    '''
    Truncate an array between set values
    '''
    filter1 = (array_x > min_val) & (array_x < max_val)
    array_x_out = array_x[filter1]
    array_y_out = array_y[filter1]
    return array_x_out, array_y_out

def truncate_x_array(array_x,min_val,max_val):
    '''
    Truncate an array between set values
    '''
    filter1 = (array_x > min_val) & (array_x < max_val)
    array_x_out = array_x[filter1]
    return array_x_out

def snv_normalisation(x_values):
    """
    Standard normal variate normalisation  
    """
    SNV = snv()
    y_values = SNV.fit_transform(x_values)
    return y_values
