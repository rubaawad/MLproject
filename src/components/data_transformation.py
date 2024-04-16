# Import necessary libraries and modules
import sys
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

# Define a dataclass to hold configuration parameters for data transformation
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

# Define a class for data transformation
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for defining the data transformation pipeline.
        '''
        try:
            # Define numerical columns and columns to impute missing values
            numerical_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
            columns_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

            # Define pipeline for imputing missing values
            impute_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="mean"))
                ]
            )

            # Define pipeline for numerical feature scaling
            num_pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler())
                ]
            )

            logging.info(f"Columns to impute: {columns_to_impute}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Define ColumnTransformer to apply different transformations to different columns
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("impute_pipeline", impute_pipeline, columns_to_impute)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        '''
        This function initiates the data transformation process.
        '''
        try:
            # Read train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            # Get data transformer object
            preprocessing_obj = self.get_data_transformer_object()
            #Get Outcome as target variable
            target_column_name = "Outcome"
            # For the training dataset:
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)  # Extract input features by dropping the target column
            target_feature_train_df = train_df[target_column_name]  # Extract target variable

            # For the testing dataset:
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)  # Extract input features by dropping the target column
            target_feature_test_df = test_df[target_column_name]  # Extract target variable

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            # Apply preprocessing object on training and testing dataframes
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Concatenate input features and target features
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object.")

            # Save preprocessing object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)