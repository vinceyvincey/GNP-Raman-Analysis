'''Script to run pipeline'''

# Script setup
# ======================================================================
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from evaluating import generic_model_evaluation
from importing import get_structured_data, prepare_data_for_learning,PrepConfig,ImportConfig, preprocess_data
from training import  generic_cross_validator,TrainingConfig
from pipeline import rf_pipeline

# Setup Program
# ======================================================================


def wrapper_rf_pipeline(
    import_params:dict, 
    prep_params:dict,
    training_params: dict):

    import_config_defined = ImportConfig(**import_params)
    prep_config_defined  = PrepConfig(**prep_params)
    training_config_defined = TrainingConfig(**training_params)

    rf_pipeline(
    
        # Importing
        # -----------------------------------------------------------------

        import_data = get_structured_data(config = import_config_defined),
        preprocess_data  = preprocess_data(config = prep_config_defined),
        prepare_data = prepare_data_for_learning(config = prep_config_defined),      
            
        # Training
        # -----------------------------------------------------------------

        model_training = generic_cross_validator(config = training_config_defined),

        # Evaluating
        # -----------------------------------------------------------------

        model_evaluation = generic_model_evaluation(config = prep_config_defined)
        
    ).run()

def main(import_params:dict, prep_params:dict,training_params:dict):
    
    wrapper_rf_pipeline(
        import_params = import_params,
        prep_params = prep_params,
        training_params = training_params,)
        
    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri {get_tracking_uri()}\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the `mlflow_example_pipeline`"
        "experiment. Here you'll also be able to compare the two runs.)"
    )

# Run Program
# ======================================================================

if __name__ == "__main__":
    
    
    features_in = [
        'Par_D_fwhm',
        'Par_D_amplitude',
        'Par_D_center',
        'Par_D_height',
        
        'Par_G_fwhm',
        'Par_G_amplitude',
        'Par_G_center',
        'Par_G_height',
        
        'Par_D2_fwhm',
        'Par_D2_amplitude', 
        'Par_D2_center',
        'Par_D2_height',

        'Par_DG_IDG',
        'Par_DG_ADG',
        'Par_D2G_IDG',
        'Par_D2G_ADG',
        'sigma',
        'La2',
        ]

    features_in = ['Par_D_fwhm', 'Par_G_fwhm', 'Par_G_amplitude', 'Par_G_center',
       'Par_D2_fwhm', 'Par_D2G_IDG', 'sigma', 'La2']
    
    model_used  = 'RandomForestClassifier'

    import_params = {
        'filename':'fluorination_different_gnps_disentangled.csv'
    }
    
    prep_params = {
        'label' : 'functionalisation',
        'features' : features_in,
        'materials':['EC-GNP'],
        'functionalisations'  : ['BL','CF4'],
        'resample_groups' : ['material','functionalisation'],
        'resample_spectra': 1000,
    }
    
    training_params = {
        'training_classifier': model_used
    }
    
    main(
        import_params = import_params,
        prep_params = prep_params,
        training_params = training_params,
        )