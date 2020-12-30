from flask import Flask,render_template,request
import joblib
import numpy as np
import warnings
warnings.filterwarnings("ignore")

model=joblib.load('heart_risk_prediction_regression_model.sav')

app = Flask(__name__)

@app.route('/',methods=['GET'])
def index():
    return render_template("patient_details.html")

@app.route('/getresults',methods=['GET','POST'])
def results():
    
    result=request.form 
    name=result['name']
    gender=float(result['gender'])
    age=float(result['age'])
    tc=float(result['tc'])
    hdl=float(result['hdl']) 
    sbp=float(result['sbp']) 
    smoke=float(result['smoke'])
    bpm=float(result['bpm'])
    diab=float(result['diab'])
    
    test_data=np.array([gender,age,tc,hdl,smoke,bpm,diab]).reshape(-1,1)
    
    prediction=model.predict(test_data) 
    prediction = max(prediction,0)
    
    resultDict={"name":name,"risk":round(prediction[0][0],2)}
    
    return render_template('patient_results.html',results=resultDict)

if __name__ == "__main__":
    app.run(debug=True)