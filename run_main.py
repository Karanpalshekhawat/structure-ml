"""
This script is the main file that calls all the other
scripts to run the ml project.
"""

import joblib
import pandas as pd

from sklearn import metrics, tree
from src.create_folds import create_folds_using_kfold



def run_output(fold, df):
    """
    Structure, train and save the model
    for given fold number.

    Args:
        fold (int): number for fold
        df (pd.DataFrame): training dataset

    Returns:

    """
    df_train = df[df['kfold'] != fold].reset_index(drop=True)
    df_valid = df[df['kfold'] == fold].reset_index(drop=True)

    """Convert training dataframe to numpy values to use training modules"""
    x_train = df_train.drop('label', axis=1).values
    y_train = df_train['label'].values

    """Convert validation dataframe to numpy values for evaluation"""
    x_valid = df_valid.drop('label', axis=1).values
    y_valid = df_valid['label'].values






if __name__ == '__main__':
    df = create_folds_using_kfold()
