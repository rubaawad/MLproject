from flask import Flask, request, render_template, redirect
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

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
        data = CustomData(
            Pregnancies=request.form.get('Pregnancies'),
            Glucose=request.form.get('Glucose'),
            BloodPressure=request.form.get('BloodPressure'),
            SkinThickness=request.form.get('SkinThickness'),
            Insulin=request.form.get('Insulin'),
            BMI=request.form.get('BMI'),
            DiabetesPedigreeFunction=request.form.get('DiabetesPedigreeFunction'),
            Age=request.form.get('Age')
        )
        pred_df = data.get_data_as_data_frame()  # Get data as DataFrame
        print(pred_df)
        print("Before Prediction")

        predict_pipeline = PredictPipeline()  # Create PredictPipeline object
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)  # Perform prediction
        print("after Prediction", results)
        
        # Translate prediction to human-readable format
        prediction_text = "The person is likely to have Diabetes in the next 5 years" if results[0] == 1 else "The person is NOT likely to have Diabetes in the next 5 years"
        
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
    app.run(debug=True, host="0.0.0.0", port=5000)  # Run the Flask application