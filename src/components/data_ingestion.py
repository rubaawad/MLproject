# Import necessary libraries and modules
import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig

# Define a dataclass to hold configuration parameters for data ingestion
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

# Define a class for data ingestion
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Read the dataset into a DataFrame
            df = pd.read_csv('src/components/notebook/data/diabetes-2-1.csv', delimiter=',')
            logging.info('Read the dataset as dataframe')

            # Create necessary directories
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw data to a CSV file
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            # Perform train-test split
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=0)

            # Save train and test sets to CSV files
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            # Raise custom exception if an error occurs
            raise CustomException(e, sys)

# Entry point of the script
if __name__ == "__main__":
    # Instantiate DataIngestion class
    obj = DataIngestion()
    # Initiate data ingestion process
    train_data, test_data = obj.initiate_data_ingestion()

    # Instantiate DataTransformation class
    data_transformation = DataTransformation()
    # Initiate data transformation process
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    # Instantiate ModelTrainer class
    modeltrainer = ModelTrainer()
    # Initiate model training process and print results
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))