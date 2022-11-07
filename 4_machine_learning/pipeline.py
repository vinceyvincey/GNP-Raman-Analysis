"""Define zenpipelines"""

from zenml.integrations.constants import MLFLOW,TENSORFLOW
from zenml.pipelines import pipeline
from zenml.config import DockerSettings

docker_settings = DockerSettings(required_integrations=[MLFLOW])
@pipeline(enable_cache=True, settings={"docker": docker_settings})
def rf_pipeline(
    
    import_data,
    preprocess_data,
    prepare_data,
    model_training,
    model_evaluation,
    
    ):
    """ Pipeline including feature selection and hyperparameter tuning steps. """

    data = import_data()
    data_filled = preprocess_data(data = data)
    x_train, x_test, y_train, y_test, classes, all_features = prepare_data(data_filled = data_filled)
    model = model_training(x_train = x_train, y_train = y_train)

    model_evaluation(x_train = x_train, 
                     y_train = y_train, 
                     x_test = x_test, 
                     y_test = y_test, 
                     model = model, 
                     classes = classes,
                     features_select = all_features)

docker_settings = DockerSettings(required_integrations=[MLFLOW, TENSORFLOW])
@pipeline(enable_cache=True, settings={"docker": docker_settings})
def cnn_pipeline(
    import_data,
    preprocess_data,
    prepare_data,
    reshape_data,
    model_training,
    model_evaluation,
):
    """ CNN Pipeline built on TensorFlow. """

    data = import_data()
    data_filled = preprocess_data(data = data)
    x_train, x_test, y_train, y_test, classes, all_features = prepare_data(data_filled = data_filled)
    y_train, y_test  = reshape_data(y_train = y_train, y_test=y_test, classes = classes)
    model = model_training(x_train = x_train, y_train = y_train)
    model_evaluation(x_train = x_train,
                     y_train = y_train,
                     x_test = x_test,
                     y_test = y_test,
                     model = model,
                     classes = classes,
                     features_select = all_features)