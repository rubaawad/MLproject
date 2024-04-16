import sys 
import os 
import pandas as pd 
from src.exception import CustomException 
from src.utils import load_object 

class PredictPipeline:
    def __init__(self):
        pass  # Initializing the PredictPipeline class

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")  # Path to the trained model file
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')  # Path to the preprocessor file

            print("Before Loading")  # Print statement before loading the model and preprocessor
            model = load_object(file_path=model_path)  # Loading the trained model
            preprocessor = load_object(file_path=preprocessor_path)  # Loading the preprocessor
            print("After Loading")  # Print statement after loading the model and preprocessor
            
            data_scaled = preprocessor.transform(features)  # Scaling the input features using the preprocessor
            print("data scaled", data_scaled)  # Print the scaled data
            
            preds = model.predict(data_scaled)  # Making predictions using the model
            return preds  # Returning the predictions
        
        except Exception as e:
            raise CustomException(e, sys)  # Raise CustomException if an error occurs

class CustomData:
    def __init__(
        self,
        Pregnancies: float,
        Glucose: float,
        BloodPressure: float,
        SkinThickness: float,
        Insulin: float,
        BMI: float,
        DiabetesPedigreeFunction: float,
        Age: float
    ):
        print("in CustomData")  # Print statement indicating initialization

        # Initializing class attributes with provided values
        self.Pregnancies = Pregnancies
        self.Glucose = Glucose
        self.BloodPressure = BloodPressure
        self.SkinThickness = SkinThickness
        self.Insulin = Insulin
        self.BMI = BMI
        self.DiabetesPedigreeFunction = DiabetesPedigreeFunction
        self.Age = Age

    def get_data_as_data_frame(self):
        print(" in get_data_as_data_frame")  # Print statement indicating the function call
        try:
            # Creating a dictionary with custom data
            custom_data_input_dict = {
                "Pregnancies": [self.Pregnancies],
                "Glucose": [self.Glucose],
                "BloodPressure": [self.BloodPressure],
                "SkinThickness": [self.SkinThickness],
                "Insulin": [self.Insulin],
                "BMI": [self.BMI],
                "DiabetesPedigreeFunction": [self.DiabetesPedigreeFunction],
                "Age": [self.Age],
            }

            # Converting the dictionary to a pandas DataFrame
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)  # Raise CustomException if an error occurs
