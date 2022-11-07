'''Script to run pipeline'''

# Setup Program
# ======================================================================

import numpy as np
from zenml.environment import Environment

from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri


from zenml.client import Client
experiment_tracker = Client().active_stack.experiment_tracker

from pipeline import cnn_pipeline
from importing import ImportConfig, PrepConfig, get_spectral_data, preprocess_data, prepare_data_for_learning, reshape_y_sets
from training import CNNConfig, cnn_trainer
from evaluating import cnn_model_evaluation

def wrapper_cnn_pipeline(
    import_params:dict, 
    prep_params:dict,
    training_params: dict):
    
    import_config_defined   = ImportConfig(**import_params)
    prep_config_defined     = PrepConfig(**prep_params)
    cnn_config_defined      = CNNConfig(**training_params)
    
    cnn_pipeline(
        
        # Importing
        # -----------------------------------------------------------------
        
        import_data         = get_spectral_data(config = import_config_defined),
        preprocess_data     = preprocess_data(config = prep_config_defined),
        prepare_data        = prepare_data_for_learning(config = prep_config_defined),
        reshape_data        = reshape_y_sets(),
        
        # Training
        # -----------------------------------------------------------------
        
        model_training = cnn_trainer(config = cnn_config_defined),
        
        # Evaluating
        # -----------------------------------------------------------------
        
        model_evaluation = cnn_model_evaluation(config = prep_config_defined)
        
    ).run()

def main(
    import_params:dict, 
    prep_params:dict,
    training_params: dict):
    
    wrapper_cnn_pipeline(
        import_params   = import_params, 
        prep_params     = prep_params,
        training_params = training_params,
    )
    
    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri {get_tracking_uri}\n"
        "To inspect your experiment runs within the mlflow ui.\n"
        "You can find your runs tracked within the `mlflow_example_pipeline`"
        "experiment. Here you'll also be able to compare the two runs.)"
    )
    

# Run Program
# ======================================================================

if __name__ == "__main__":
    
    import_params = {
        'filename':'puregraph_size_comparison_X.csv'
    }
    
    prep_params = {
        'label'             : 'material',
        'features'          : list(str(int(a)) for a in np.concatenate([np.linspace(0,600,601),np.linspace(1200,1899,700)])),
        'materials'         : ['PG5','PG10','PG20'],
        'functionalisations': ['BL'],
        'resample_groups'   : ['material','functionalisation'],
        'resample_spectra'  : 1100,
    }
    
    training_params = {
        'epochs'    : 30,
        'batch_size': 550,
        'labels'    : 3
    }
    
    main(
        import_params   = import_params, 
        prep_params     = prep_params,
        training_params = training_params
    )