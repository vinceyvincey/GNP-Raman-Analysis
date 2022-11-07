'''Module for Carrying out Training'''

# Module setup
# ======================================================================

import numpy as np
import mlflow

from sklearn.base import ClassifierMixin

# from zenml.integrations.mlflow.mlflow_step_decorator import enable_mlflow
from zenml.steps import step, BaseParameters, Output

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Conv1D, MaxPooling1D
from keras.callbacks import ModelCheckpoint

from zenml.client import Client
experiment_tracker = Client().active_stack.experiment_tracker

# Random Forest Classifier
# ======================================================================

class TrainingConfig(BaseParameters):
    training_classifier: str 

@step(enable_cache=False, experiment_tracker=experiment_tracker.name)
def generic_cross_validator(
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: TrainingConfig) -> Output(model = ClassifierMixin):

    """ Carrying out cross validation using tuned parameters. """
    
    classifier_dict = {'RandomForestClassifier':RandomForestClassifier,
                       'GradientBoostingClassifier':GradientBoostingClassifier,
                       'XGBClassifier':XGBClassifier}
    
    model = classifier_dict[config.training_classifier](**{'random_state':1})

    # Log TrainingConfig parameters
    config_dict = vars(config)
    mlflow.log_params(config_dict)

    # Log the tuned parameters to 
    for key, value in model.get_params().items():
        mlflow.log_param(key, value)

    kfold_scores = cross_val_score(model, x_train,y_train,cv=StratifiedKFold(5))

    mlflow.log_metric(f"kfold_average_accuracy", kfold_scores.mean())
    mlflow.log_metric(f"kfold_std_accuracy", kfold_scores.std())

    return model

# CNN Classifier
# ======================================================================

class CNNConfig(BaseParameters):
    epochs: int
    batch_size:int
    labels:int

@step(enable_cache=True, experiment_tracker=experiment_tracker.name)
def cnn_trainer(
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: CNNConfig,
)-> Output(model = Model):
    
    print(f'x train shape: {x_train.shape}, y train shape: {y_train.shape}')

    model = Sequential()
    model.add(Reshape((x_train.shape[1], 1), input_shape=(x_train.shape[1],)))
    
    model.add(Conv1D(filters = 64, kernel_size=3, strides=1, activation='relu',padding='valid'))
    model.add(MaxPooling1D(pool_size=2,strides=2, padding = 'valid'))
    
    model.add(Conv1D(filters = 32, kernel_size=3, strides=1, activation='relu',padding='same'))
    model.add(MaxPooling1D(pool_size=2,strides=2, padding = 'valid'))
    
    model.add(Dropout(0.3)) # 0.2
    
    model.add(Flatten())
    
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(config.labels, activation='softmax'))
    
    print(model.summary())
    
    mlflow.tensorflow.autolog()
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
    
    callbacks_list = [ModelCheckpoint(filepath='data/output/models/best_model.{epoch:02d}-{val_loss:.2f}.h5',
                                      monitor='val_loss', 
                                      save_best_only=True)]
    
    batch_size = config.batch_size
    epochs = config.epochs

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks_list,
        class_weight = {
            0:1.5,
            1:2.2,
            2:0.8},
        validation_split=0.1,
        verbose=1)

    return model