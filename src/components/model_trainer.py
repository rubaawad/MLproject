import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pandas as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self, train_array, test_array):
        """
        Train and evaluate machine learning models based on the input data.

        Parameters:
            train_array (numpy.ndarray): The training dataset.
            test_array (numpy.ndarray): The testing dataset.

        Returns:
            float: The accuracy score of the best model on the testing dataset.

        Raises:
            CustomException: If an error occurs during model training and evaluation.
        """
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
            train_array[:, :-1],
            train_array[:, -1],
            test_array[:, :-1],
            test_array[:, -1]
            )
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),  # Increase max_iter
                "Knn": KNeighborsClassifier(),
                "svm": SVC(),
                "Naive Bayes": GaussianNB(),
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
            }

            # Define parameters for grid search
            params = {
                "Decision Tree": {
                    'max_depth': [None, 10, 20, 30, 40],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                },
                "Random Forest": {
                    'n_estimators': [100, 200],
                    'max_depth': [None, 10],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                    'max_features': ['sqrt', 'log2']
                },
                "svm": {
                        'kernel': ['linear', 'rbf'],
                },
                "Naive Bayes": {},
                "Knn": {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance']

                },

                "Logistic Regression": {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'penalty': ['l2']
                }
            }

            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                             models=models,param=params)

            # Now, iterate over the models in model_report to find the best model
            best_model_name = None
            best_model_score = float('-inf')

            for model_name, metrics in model_report.items():
            # Compute a score for each model (e.g., average of accuracy, precision, and F1-score)
                score = (metrics['test_model_accuracy'] + metrics['test_model_precision'] + metrics['test_model_f1']) / 3
    
            # Check if this model has a higher score than the current best model
                if score > best_model_score:
                    best_model_score = score
                    best_model_name = model_name
            best_model = models[best_model_name]
            print("bestmodel is ",best_model_name)
            #if best_model_score < 0.6:
             #   raise CustomException("No best model found")
              #  logging.info(f"Best found model on both training and testing dataset")

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Predict using the best model
            predicted = best_model.predict(X_test)

            # Calculate accuracy score
            score = accuracy_score(y_test, predicted) * 100
            return score
            
        except Exception as e:
            raise CustomException(e,sys)