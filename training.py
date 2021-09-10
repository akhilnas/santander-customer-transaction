### Required Libraries ###

# Handle Arguments for File
import argparse
import os

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

# Plotting Tools
import matplotlib.pyplot as plt
from xgboost import plot_importance

# MLflow
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import mlflow.xgboost

import logging
import pickle


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def train(X_train, X_valid, y_train, y_valid, hyperparameter = False, *args):
    
    ### Model Setup & Training ###    
    
    # Set up Parameters of Training    
    for ar in args:            
        opt = ar              

    # GPU Parameter
    device_method = 'gpu_hist' if (len(GPUtil.getAvailable()) != 0) else 'auto'
    
    # Model instantiation
    
    # Set an experiment name, which must be unique and case sensitive.
    mlflow.set_experiment("Santander XGBoost")
    
    with mlflow.start_run():
        model_xgboost = xgboost.XGBClassifier(eval_metric='auc',
                                            use_label_encoder=False,
                                            tree_method = device_method,
                                            verbosity=1, **opt)
        # Validation Set
        eval_set = [(X_valid, y_valid)]

        # Creating the DMatrix
        d_matrix = xgboost.DMatrix(data=X_train, label=y_train)

        xgb_param = model_xgboost.get_xgb_params()

        # Number of Cross Validation Folds
        cv_folds = 10
        early_stopping_rounds = 10
        # Cross-validation with 10 folds
        cvresult = xgboost.cv(xgb_param, d_matrix, num_boost_round=model_xgboost.get_params()['n_estimators'], 
                    nfold=cv_folds, metrics='auc', early_stopping_rounds=early_stopping_rounds, verbose_eval=False)

        model_xgboost.set_params(n_estimators=cvresult.shape[0])
        
        # Training
        model_xgboost.fit(X_train,
                    y_train,
                    early_stopping_rounds=10,
                    eval_set=eval_set,                  
                    verbose=False)
        
        # Print Results
        if hyperparameter == False:
            print("AUC Train Mean Score: {:.4f} with Standard Deviation {:.4f}\nAUC Valid Mean Score: \
                {:.4f} with Standard Deviation {:.4f}".format(cvresult['train-auc-mean'].iloc[-1],
                                                        cvresult['train-auc-std'].iloc[-1], 
                                                        cvresult['test-auc-mean'].iloc[-1], 
                                                        cvresult['test-auc-std'].iloc[-1]))
        
        # Print Results on Test-Data
        y_train_pred = model_xgboost.predict_proba(X_train)[:,1]
        y_valid_pred = model_xgboost.predict_proba(X_valid)[:,1]
        
        # AUC Scores
        auc_train = roc_auc_score(y_train, y_train_pred)
        auc_valid = roc_auc_score(y_valid, y_valid_pred)
        

        print("AUC Train: {:.4f}\nAUC Test: {:.4f}".format(auc_train, auc_valid))     
     
        
        # MLflow Parameters
        mlflow.log_param('n_estimators', cvresult.shape[0])
        mlflow.log_param('learning_rate', opt['learning_rate'])
        mlflow.log_param('max_depth', opt['max_depth'])
        mlflow.log_param('subsample', opt['subsample'])
        mlflow.log_param('colsample_bytree', opt['colsample_bytree'])
        # MLflow Metrics
        mlflow.log_metric("AUC Train", auc_train)
        mlflow.log_metric("AUC Valid", auc_valid)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.xgboost.log_model(model_xgboost, "model", registered_model_name="XGBoostModel")
        else:
            mlflow.xgboost.log_model(model_xgboost, "model")

    # Feature Importance Plot
    if hyperparameter == False:
        ax = plot_importance(model_xgboost, max_num_features=15, importance_type='gain')  
        ax.figure.savefig('feature_importance.png')
    
        # Pickle the model
        file_name = "xgb_cls.pkl"
    
        # save
        pickle.dump(model_xgboost, open(file_name, "wb"))   
    
    return y_valid_pred
    
def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)    

if __name__=='__main__':
    
    ### Passed Arguments ###
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=int, default=0.1, help="Learning Rate of Xgboost model i.e Weightage of constructed trees")
    parser.add_argument("--max_depth", type=int, default=5, help="Maximum Depth of of each tree in the XGBoost Classifier")
    parser.add_argument("--subsample", type=float, default=0.5, help="The sampling percentage of the Training data used to create a Tree.")
    parser.add_argument("--colsample_bytree", type=float, default=0.5, help="Percentage of Features to be used while building a tree in the model.") 
    parser.add_argument("--data_dir", type=dir_path, default='/data', help="Path to Data Directory.") 
    opt = parser.parse_args()
    print(opt)

    ### Data Import ###

    # Load Data    
    data_dir = opt.data_dir
    try:
        df_train = pd.read_csv(data_dir + '/train.csv')
    except:
        logger.exception("Unable to load CSV file.")      
    
    
    # Check for GPU
    if (len(GPUtil.getAvailable()) != 0):
        print("GPU found. Running on GPU.")
    else:
        print("Running on CPU.")
    
    ### Data Preperation ###
    
    var_colums = [c for c in df_train.columns if c not in ['ID_code','target']]
    X = df_train.loc[:, var_colums]
    y = df_train.loc[:, 'target']

    # We are performing a 80-20 split for Training and Validation
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
    X_train.shape, X_valid.shape, y_train.shape, y_valid.shape
    
    ## Training and 
    # Call Training Function & Pass Training Parameters
    
    xgboost_params = {
        "learning_rate": opt.learning_rate,
        "n_estimators": 5000,
        "max_depth": opt.max_depth, 
        "subsample": opt.subsample,
        "colsample_bytree": opt.colsample_bytree,
        "random_state": 1121217
    }
    _ = train(X_train, X_valid, y_train, y_valid, False, xgboost_params) 
    
    
    
