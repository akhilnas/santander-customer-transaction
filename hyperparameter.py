### Required Libraries ###

# Handle Arguments for File
import argparse

# General Data Manipulation Libraries
import numpy as np
import pandas as pd

# Model & Helper Libraries
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
import GPUtil

# Hyper-parameter Optimization
import optuna

# Plotting Tools
import matplotlib.pyplot as plt
from xgboost import plot_importance
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_parallel_coordinate

# MLflow
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import mlflow.xgboost

import logging
import pickle

# Import Training modeule
from training import train

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def objective(trial, X_train, y_train, X_valid, y_valid):
    
    # Model Parameters to be optimized
    xgboost_params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-7, 0.3, log=True),
        "n_estimators": trial.suggest_int(name="n_estimators", low=100, high=2000, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 8), 
        "subsample": trial.suggest_categorical(name="subsample", choices=[0.4, 0.5, 0.6]),
        "colsample_bytree": trial.suggest_categorical(name="colsample_bytree", choices=[0.4, 0.5, 0.6]),
        "random_state": 1121217
    }
    
    # Call Training Function
    y_valid_pred = train(X_train, X_valid, y_train, y_valid, True, xgboost_params)   
    
    # Optimization Metric    
    return roc_auc_score(y_valid, y_valid_pred)

if __name__=='__main__':
    
    ### Data Import ###

    # Load Data
    data_dir = './data'  
    try:
        df_train = pd.read_csv(data_dir + '/train.csv')
    except:
        logger.exception("Unable to load CSV file.")      
    
    
    # Check for GPU
    if (len(GPUtil.getAvailable()) != 0):
        print("GPU found. Running on GPU.")
    
    ### Data Preperation ###
    
    var_colums = [c for c in df_train.columns if c not in ['ID_code','target']]
    X = df_train.loc[:, var_colums]
    y = df_train.loc[:, 'target']

    # We are performing a 80-20 split for Training and Validation
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
    X_train.shape, X_valid.shape, y_train.shape, y_valid.shape
    
    # Create Study Object for Optuna
    study = optuna.create_study(direction="maximize")
    # Optimize
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_valid, y_valid), n_trials=100)
    
    print(f"Optimized roc_auc_score: {study.best_value:.5f}")
    
    print("Best params:")

    for key, value in study.best_params.items():
        print(f"\t{key}: {value}")
    
    # Plots of Hyperparamter Tuning Results
    if (optuna.visualization.is_available()):
        fig1 = plot_optimization_history(study)
        fig1.write_image('optimization_history.png')
        fig2 = plot_intermediate_values(study)
        fig2.write_image('intermediate_values.png')
        fig3 = plot_parallel_coordinate(study)
        fig3.write_image('parallel_coordinate.png')
        fig4 = plot_param_importances(study)
        fig4.write_image('param_importances.png')
        
    
    