from flask import Flask, render_template, request,redirect
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

class CustomData:
    def __init__(self,
                 Pregnancies: float,
                 Glucose: float,
                 BloodPressure: float,
                 SkinThickness: float,
                 Insulin: float,
                 BMI: float,
                 DiabetesPedigreeFunction:float,
                 Age:float
                ):
        print("in CustomData")

        self.Pregnancies = Pregnancies

        self.Glucose = Glucose

        self.BloodPressure = BloodPressure

        self.SkinThickness = SkinThickness

        self.Insulin = Insulin

        self.BMI = BMI

        self.DiabetesPedigreeFunction = DiabetesPedigreeFunction

        self.Age = Age

app = Flask(__name__)

# Load your trained model
model = joblib.load('best_model.pkl')

# Load your fitted StandardScaler
scaler = joblib.load('scaler.pkl')

def make_predictions(scaler, input_data):
    # Check if input_data has the correct format and dimensions
    if not isinstance(input_data, (tuple, list)):
        raise ValueError("Input data must be a tuple or a list.")
    
    # Convert input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the array as we are predicting one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Standardize the input data
    std_data = scaler.transform(input_data_reshaped)

    # Make predictions using the model
    predictions = model.predict(std_data)
    
    return predictions

@app.route('/')
def index():
    return redirect('/diabetes')  # Redirect to diabetes prediction page


@app.route('/diabetes', methods=['GET', 'POST'])
def predict_datapoint():
    print("in predict_datapoint")
    if request.method == 'GET':
        return render_template('index.html', form_data={})  # Render the form template with empty form data
    else:
        print("call CustomData")
        # Create CustomData object with form data
        Pregnancies=float(request.form.get('Pregnancies'))
        Glucose=float(request.form.get('Glucose'))
        BloodPressure=float(request.form.get('BloodPressure'))
        SkinThickness=float(request.form.get('SkinThickness'))
        Insulin=float(request.form.get('Insulin'))
        BMI=float(request.form.get('BMI'))
        DiabetesPedigreeFunction=float(request.form.get('DiabetesPedigreeFunction'))
        Age=float(request.form.get('Age'))     
        data = (Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age)
        # Perform prediction
        prediction = make_predictions(scaler, data)
        print('predictions are: ', prediction)
        # Translate prediction to human-readable format
        prediction_text = "The person is likely to have Diabetes in the next 5 years" if prediction[0] == 1 else "The person is NOT likely to have Diabetes in the next 5 years"
        
        # Pass form data back to the template
        form_data = {
            'Pregnancies': request.form.get('Pregnancies'),
            'Glucose': request.form.get('Glucose'),
            'BloodPressure': request.form.get('BloodPressure'),
            'SkinThickness': request.form.get('SkinThickness'),
            'Insulin': request.form.get('Insulin'),
            'BMI': request.form.get('BMI'),
            'DiabetesPedigreeFunction': request.form.get('DiabetesPedigreeFunction'),
            'Age': request.form.get('Age')
        }

        # Render the template with prediction result and form data
        return render_template('index.html', prediction=prediction_text, form_data=form_data)



if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)