# (4) Machine Learning

Initialise zenml repository and install the mlflow extentions
```bash
# Install integrations
zenml integration install mlflow tensorflow

# Initialise zenml
zenml init

# Add experiment tracker stack component
zenml experiment-tracker register mlflow_experiment_tracker --flavor=mlflow

# Register a new  stack  with experiment tracker component
zenml stack register mlflow_stack  -a default  -o default  -e mlflow_experiment_tracker

# Activate the new stack
zenml stack set mlflow_stack
```

1. To use the random forest classifier, specify the run conditions [here](run_random_forest.py), then:

```python
python run_random_forest.py
```

2. To use the cnn, specify the run conditions [here](run_cnn.py), then:

```python
python run_cnn.py
```
