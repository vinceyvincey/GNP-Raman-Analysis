''' Module for Evaluating Training Results'''

# Module setup
# ======================================================================

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import mlflow
from mlflow.models.signature import infer_signature

from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay,precision_recall_curve,PrecisionRecallDisplay
from sklearn.inspection import permutation_importance

# from zenml.integrations.mlflow.mlflow_step_decorator import enable_mlflow
from zenml.steps import step, Output

from keras.models import Model

from importing import PrepConfig

from zenml.client import Client
experiment_tracker = Client().active_stack.experiment_tracker

# Evaluate Random Forest
# ======================================================================

# Support Functions
# ----------------------------------------------------------------------

def calculate_key_metrics(y_train,y_test,y_pred_test,y_pred_train):
    """ Calculate key metrics """

    # Accuracy
    mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_pred_test))
    mlflow.log_metric("training_accuracy", accuracy_score(y_train, y_pred_train))

    # F1
    mlflow.log_metric("test_f1", f1_score(y_test, y_pred_test,average='macro'))
    mlflow.log_metric("train_f1", f1_score(y_train, y_pred_train,average='macro'))

    # Precision
    mlflow.log_metric("test_f1", f1_score(y_test, y_pred_test,average='macro'))
    mlflow.log_metric("train_f1", f1_score(y_train, y_pred_train,average='macro'))
    

def plot_confusion_matrix(y_test,y_pred_test,classes):
    
    """ Plot a confusion matrix. """
    cm = confusion_matrix(y_test, y_pred_test,normalize='true')
    fig = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig.plot(cmap=plt.cm.Blues)
    fig.im_.set_clim(0, 1)
    figloc = Path.cwd().joinpath('figures','confusion_matrix.png')
    plt.savefig(figloc,dpi=300,transparent=True)
    mlflow.log_artifact(figloc)
    
    csvloc = Path.cwd().joinpath('data','output','confusion_matrix.csv')
    cm_out = pd.DataFrame(cm)
    cm_out.to_csv(csvloc)
    mlflow.log_artifact(csvloc)
    
    plt.close("all")

def plot_roc_curve(y_test,y_score_test,features):
    """ Plot  ROC curve. """
    
    # Binary Classification
    if len(features) == 2:
    
        fpr, tpr, _ = roc_curve(y_test, y_score_test[:,1])
        fig = RocCurveDisplay(fpr=fpr, tpr=tpr)
        fig.plot()
        figloc = Path.cwd().joinpath('figures','roc_curve.png')
        plt.savefig(figloc,dpi=300)
        mlflow.log_artifact(figloc)
        plt.close("all")

def plot_precision_recall_curve(y_test,y_score_test, features):
    """ Plot Precision Recall curve. """
    
    # Binary Classification
    if len(features) == 2:

        prec, recall, _ = precision_recall_curve(y_test, y_score_test[:,1])
        fig = PrecisionRecallDisplay(precision=prec, recall=recall)
        fig.plot()
        figloc = Path.cwd().joinpath('figures','precision_recall_curve.png')
        plt.savefig(figloc,dpi=300)
        mlflow.log_artifact(figloc)
        plt.close("all")
        
def plot_feature_importance(model,x_test,y_test,feature_names):
    """ Plot the feature importance. """

    # Feature Importance (Ordered by Importance)
    
    result = permutation_importance(model, x_test, y_test, n_repeats=10, random_state=0, n_jobs=2)
    forest_importances = pd.DataFrame(result.importances_mean, index=feature_names,columns = ['feature_permutation_importance']).reset_index(drop=False)
    forest_importances = forest_importances.sort_values(by = 'feature_permutation_importance')
    
    
    csvloc = Path.cwd().joinpath('data','output','forest_importances.csv')
    forest_importances.to_csv(csvloc)
    mlflow.log_artifact(csvloc)

    graph = sns.barplot(
        data = forest_importances,
        x = 'index',
        y= 'feature_permutation_importance')
    
    graph.set_xticklabels(graph.get_xticklabels(),rotation = 90)
    plt.tight_layout()
    figloc = Path.cwd().joinpath('figures','forest_importances.png')
    plt.savefig(figloc,dpi=300)
    mlflow.log_artifact(figloc)
    plt.close("all")

# Pipeline Step
# ----------------------------------------------------------------------

@step(enable_cache=True, experiment_tracker=experiment_tracker.name)
def generic_model_evaluation(
    x_train:np.ndarray,
    y_train:np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    model: ClassifierMixin,
    classes: list,
    features_select: list,
    config: PrepConfig) -> None:

    model.fit(x_train, y_train)

    y_pred_test = model.predict(x_test)
    y_pred_train = model.predict(x_train)
    y_score_test = model.predict_proba(x_test)
    
    signature = infer_signature(x_train, model.predict(x_train))
    mlflow.sklearn.log_model(model, "model", signature=signature)
    
    calculate_key_metrics(y_train,y_test,y_pred_test,y_pred_train)
    plot_confusion_matrix(y_test,y_pred_test,classes)
    plot_precision_recall_curve(y_test,y_score_test, config.functionalisations)
    plot_roc_curve(y_test,y_score_test, config.functionalisations)
    plot_feature_importance(model,x_train,y_train,features_select)

@step(enable_cache=True, experiment_tracker=experiment_tracker.name)
def cnn_model_evaluation(
    x_train:np.ndarray,
    y_train:np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    model: Model,
    classes: list,
    features_select: list,
    config: PrepConfig) -> None:
    
    classes_dict = {
        'PG5':'PG5',
        'PG10':'PG10',
        'PG20':'PG20',
    }
    
    classes = [classes_dict[a] for a in classes]
    
    y_pred_test = np.argmax(model.predict(x_test), axis=-1)
    y_pred_train = np.argmax(model.predict(x_train), axis=-1)
    y_train = np.argmax(y_train,axis=-1)
    y_test = np.argmax(y_test,axis=-1)
    
    calculate_key_metrics(y_train,y_test,y_pred_test,y_pred_train)
    plot_confusion_matrix(y_test,y_pred_test,classes)
    