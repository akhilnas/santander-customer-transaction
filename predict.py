### Required Libraries ###

import os

# General Data Manipulation Libraries
import numpy as np
import pandas as pd

# Log and Pickle Libraries
import logging
import pickle

# Model & Helper Libraries
import xgboost
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

if __name__=='__main__':
    
    ### Model & Data Import ###
    
    # Load model
    file_name = "xgb_cls.pkl"
    xgb_model_loaded = pickle.load(open(file_name, "rb"))    
    
    # Load Data
    data_dir = './data'  
    try:
        df_test = pd.read_csv(data_dir + '/test.csv')
    except:
        logger.exception("Unable to load CSV file.")
    
    
    ### Data Preperation ###
    
    var_colums = [c for c in df_test.columns if c not in ['ID_code']]
    X = df_test.loc[:, var_colums]    
    
    ### Model Prediction ###
    
    y_test_pred = xgb_model_loaded.predict(X)
    
    ### Save Results ###
    
    # Construct Dataframe 
    data = {'ID_code':df_test.loc[:, 'ID_code'],'Target':y_test_pred}
    df = pd.DataFrame(data)
    
    # Create New Datafile    
    filename =  'predict.csv'
    if os.path.isfile(filename):
        print('File Exists. Going to overwrite.')
        # Writing Data to csv file
        df.to_csv(filename, index = False, header=True)
    else:
        # Writing Data to csv file
        df.to_csv(filename, index = False, header=True)
    
    