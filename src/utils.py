import os
import sys

import numpy as np 
import pandas as pd
#import dill
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_object(file_path, obj):
    """
    Save a Python object to a file using pickle serialization.

    Parameters:
        file_path (str): The file path where the object will be saved.
        obj: The Python object to be saved.

    Raises:
        CustomException: If an error occurs while saving the object.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)  # Create directory if it doesn't exist
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)  # Serialize and save the object to the file
    except Exception as e:
        raise CustomException(e, sys)  # Raise CustomException if an error occurs

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Train models, perform grid search, and evaluate their performance.

    Parameters:
        X_train (array-like): Training input features.
        y_train (array-like): Training target labels.
        X_test (array-like): Testing input features.
        y_test (array-like): Testing target labels.
        models (dict): Dictionary containing the models to be evaluated.
        param (dict): Dictionary containing the parameter grids for grid search.

    Returns:
        dict: Dictionary containing evaluation metrics for each model.

    Raises:
        CustomException: If an error occurs during model training and evaluation.
    """
    try:
        report = {}  # Initialize an empty dictionary to store evaluation metrics
        
        # Iterate over each model in the dictionary
        for i in range(len(list(models))):
            model = list(models.values())[i]  # Get the model
            para = param[list(models.keys())[i]]  # Get the parameter grid for grid search

            # Perform grid search for hyperparameter tuning
            grid_search = GridSearchCV(estimator=model, param_grid=para, cv=5)
            grid_search.fit(X_train, y_train)  # Fit the grid search to the training data
            
            # Get the best parameters and best score
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
    
            print(f"Best Parameters for {model.__class__.__name__}: {best_params}")  # Print the best parameters
            print(f"Best Score for {model.__class__.__name__}: {best_score}\n")  # Print the best score
            
            # Set the model with the best parameters and fit it to the training data
            model.set_params(**grid_search.best_params_)
            model.fit(X_train, y_train)

            # Make predictions on the test set
            y_test_pred = model.predict(X_test)
            
            # Compute evaluation metrics
            test_model_accuracy = accuracy_score(y_test, y_test_pred)
            test_model_precision = precision_score(y_test, y_test_pred)
            test_model_recall = recall_score(y_test, y_test_pred)
            test_model_f1 = f1_score(y_test, y_test_pred)
            test_model_roc_auc = roc_auc_score(y_test, y_test_pred)
    
            # Compute cross-validation accuracy
            test_model_cv_accuracy = cross_val_score(model, X_test, y_test, cv=5, scoring='accuracy').mean()

            # Add the metrics with their names to the report dictionary
            report[list(models.keys())[i]] = {
                'test_model_cv_accuracy': test_model_cv_accuracy,
                'test_model_accuracy': test_model_accuracy,
                'test_model_precision': test_model_precision,
                'test_model_recall': test_model_recall,
                'test_model_f1': test_model_f1,
                'test_model_roc_auc': test_model_roc_auc
            }
        print(report)

        return report  # Return the evaluation report

    except Exception as e:
        raise CustomException(e, sys)  # Raise CustomException if an error occurs

def load_object(file_path):
    """
    Load a Python object from a file using pickle deserialization.

    Parameters:
        file_path (str): The file path from which the object will be loaded.

    Returns:
        The Python object loaded from the file.

    Raises:
        CustomException: If an error occurs while loading the object.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)  # Deserialize
    except Exception as e:
        raise CustomException(e, sys)  # Raise CustomException if an error occurs





