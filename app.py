from flask import Flask, request, render_template, redirect
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route('/')
def index():
    return redirect('/diabetes') 

@app.route('/diabetes', methods=['GET', 'POST'])
def predict_datapoint():
    print("in predict_datapoint")
    if request.method == 'GET':
        return render_template('index.html', form_data={})  # Pass empty form_data for initial rendering
    else:
        print("call CustomData")
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
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
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

        return render_template('index.html', prediction=prediction_text, form_data=form_data)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)