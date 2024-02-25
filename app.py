from flask import Flask,request,render_template,redirect

from src.pipeline.predict_pipeline import CustomData,PredictPipeline
import pandas as pd
import numpy as np

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    print("in / index")
    return redirect('/diabetes') 

@app.route('/diabetes',methods=['GET','POST'])
def predict_datapoint():
    print("in predict_datapoint")
    if request.method=='GET':
        return render_template('index.html')
    else:
        print("call CustomData")
        data=CustomData(

            Pregnancies= request.form.get('Pregnancies'),
            Glucose= request.form.get('Glucose'),
            BloodPressure= request.form.get('BloodPressure'),
            SkinThickness= request.form.get('SkinThickness'),
            Insulin= request.form.get('Insulin'),
            BMI= request.form.get('BMI'),
            DiabetesPedigreeFunction= request.form.get('DiabetesPedigreeFunction'),
            Age= request.form.get('Age')
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction",results)
         # Translate prediction to human-readable format
        prediction_text = "This person has Diabetes" if results[0] == 1 else "This person has No Diabetes"

        return render_template('index.html', prediction=prediction_text)
    

if __name__=="__main__":
    app.run(debug=True)
    app.run(host="0.0.0.0")        