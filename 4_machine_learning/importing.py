''' Module for importing data '''

# Module setup
# ======================================================================

from pathlib import Path
import pandas as pd
import numpy as np
import mlflow

from zenml.steps import step, Output, BaseParameters
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

from zenml.client import Client
experiment_tracker = Client().active_stack.experiment_tracker

print(experiment_tracker)

# (1) Import data file
# ======================================================================

# Pipeline Step
# ----------------------------------------------------------------------

class ImportConfig(BaseParameters):
    """ Configuration file for importing data. """
    filename: str

@step(enable_cache = False, experiment_tracker=experiment_tracker.name)
def get_structured_data(config:ImportConfig) -> Output(
    data = pd.DataFrame
):
    """Loads the .csv file containing the structured raman data"""

    data = pd.read_csv(Path.cwd().joinpath('data','input',config.filename)).iloc[:,1:]
    mlflow.log_param('filename', config.filename)

    return data

@step(enable_cache = False, experiment_tracker=experiment_tracker.name)
def get_spectral_data(config:ImportConfig) -> Output(
    data = pd.DataFrame
):
    """Loads the .csv file containing the unstructured spectral raman data"""

    data = pd.read_csv(Path.cwd().joinpath('data','input',config.filename))
    mlflow.log_param('filename', config.filename)

    return data

# (2) Prepare data for Learning
# ======================================================================

# Support Functions
# ----------------------------------------------------------------------

def filter_data(data:pd.DataFrame, materials:list, functionalisations:list):
    """Filters the material and functionalisations to include"""

    filter_in = (data['material'].isin(materials)) &\
         (data['functionalisation'].isin(functionalisations)) 
    return data.loc[filter_in]    

def labels_to_classes(data:pd.DataFrame,target_label:str):
    """Encodes categorical labels into an integer"""

    classes = data[target_label].unique().tolist()

    print(f"Label classes: {classes}")
    data[target_label] = data[target_label].map(classes.index)
    return data, classes

def sample_spectra(df_in: pd.DataFrame, group_columns: list, spectra_number: int):
    """ Sample a desired number of spectra. """
    
    df_out = pd.DataFrame()
    for _, group in df_in.groupby(group_columns):
        group_sampled = group.sample(n=spectra_number, random_state=0)
        df_out = pd.concat([df_out, group_sampled])

    df_out = df_out.reset_index(drop=True)

    return df_out

# Pipeline Step
# ----------------------------------------------------------------------

class PrepConfig(BaseParameters):
    """Data Preparation Parameters"""
    label: str
    features: list
    materials: list
    functionalisations: list
    resample_groups: list
    resample_spectra: int

@step(enable_cache=False, experiment_tracker=experiment_tracker.name)
def preprocess_data(
    data: pd.DataFrame,
    config : PrepConfig,
) -> Output(data_filled = pd.DataFrame):

    # Filter data to include only materials and functionalisations of interest
    data_filtered = filter_data(data,config.materials,config.functionalisations)

    # Sample data to include a consistent number of samples for each map
    data_sampled = sample_spectra(data_filtered, config.resample_groups, config.resample_spectra)

    # Fill missing data with zeros
    data_filled = data_sampled.fillna(0)
        
    # Print sample sizes
    print(data_filled.groupby(config.resample_groups).count().iloc[:,0])
    
    # Log PrepConfig parameters
    config_dict = vars(config)
    
    if len(str(config_dict['features'])) > 250: # workaround
        config_dict['features'] = 'exceeded' 
    
    mlflow.log_params(config_dict)
    
    return data_filled

def feature_normalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu)/sigma

@step(enable_cache=True)
def normalise_features(
    input_data: pd.DataFrame,
    config : PrepConfig,
) -> Output(output_data = pd.DataFrame):
    
    input_data[config.features] = feature_normalize(input_data[config.features])
    output_data = input_data
    
    return output_data

@step(enable_cache=True)
def prepare_data_for_learning(
    data_filled: pd.DataFrame,
    config: PrepConfig,
) -> Output(y_train = np.ndarray,x_train= np.ndarray,y_test= np.ndarray,x_test= np.ndarray,classes = list, all_features = list):
    """ Split data into training and test sets. """
    
    # Turn string labels into numbered classes
    data_labelled, classes = labels_to_classes(data_filled,config.label)

    # Split into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(data_labelled[config.features], data_labelled[config.label], test_size=0.3, random_state=1)
    x_train = x_train.values
    x_test = x_test.values
    y_train = y_train.values
    y_test = y_test.values
    
    all_features  = config.features
    
    print(f'Training Set (x): {x_train.shape}, Training Set (y): {y_train.shape}')
    print(f'Test Set (x): {x_test.shape}, Test Set (y): {y_test.shape}')
    
    return x_train, x_test, y_train, y_test, classes, all_features

@step(enable_cache=True)
def reshape_y_sets(
    y_train: np.ndarray,
    y_test:np.ndarray,
    classes:list,
) -> Output(y_train_reshape = np.ndarray,y_test_reshape= np.ndarray):
    
    y_train_reshape = np_utils.to_categorical(y_train, len(classes))
    y_test_reshape = np_utils.to_categorical(y_test, len(classes))
    
    return y_train_reshape, y_test_reshape