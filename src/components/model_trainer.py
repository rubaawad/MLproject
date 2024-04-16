import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
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
    trained_model_file_path = os.path.join("artifacts", "model.pkl")  # Configuration for the path to save the trained model

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()  # Initializing ModelTrainerConfig object

    def initiate_model_trainer(self, train_array, test_array):
        """
        Train and evaluate machine learning models based on the input data.

        Parameters:
            train_array: The training dataset.
            test_array: The testing dataset.

        Returns:
            float: The accuracy score of the best model on the testing dataset.

        Raises:
            CustomException: If an error occurs during model training and evaluation.
        """
        try:
            logging.info("Split training and test input data")  # Logging message to indicate splitting of training and test data
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )  # Splitting the input arrays into features and target variables

            # Dictionary containing different classifiers
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),  # Increasing max_iter for logistic regression
                "Knn": KNeighborsClassifier(),  # Initializing K-Nearest Neighbors Classifier
                "Naive Bayes": GaussianNB(),  # Initializing Gaussian Naive Bayes Classifier
                "Random Forest": RandomForestClassifier(),  # Initializing Random Forest Classifier
                "Decision Tree": DecisionTreeClassifier(),  # Initializing Decision Tree Classifier
                "svm": SVC()  # Initializing Support Vector Classifier
            }

            # Dictionary containing hyperparameter grids for grid search
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
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto', 0.1, 1]
                },
                "Naive Bayes": {},  # No hyperparameters for Naive Bayes
                "Knn": {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance']
                },
                "Logistic Regression": {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'penalty': ['l2']
                }
            }
            # Specify scorers for grid search
            scorers = {
                'precision_score': make_scorer(precision_score),
                'recall_score': make_scorer(recall_score),
                'accuracy_score': make_scorer(accuracy_score)
            }

            # Evaluate models using cross-validation and grid search
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                  models=models, param=params)

            # Find the best model based on the highest score
            best_model_name, best_model_score = max(model_report.items(),
                                                    key=lambda x: (x[1]['test_model_accuracy'] + x[1]['test_model_f1']) / 2)


            best_model = models[best_model_name]  # Select the best model

            logging.info(f"Best found model on both training and testing dataset: {best_model_name}")  # Log the best model

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Predict using the best model
            predicted = best_model.predict(X_test)

            # Calculate accuracy score
            score = accuracy_score(y_test, predicted) * 100
            return score  # Return the accuracy score

        except Exception as e:
            raise CustomException(e, sys)  # Raise CustomException if an error occurs during model training and evaluation