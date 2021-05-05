from flask import Flask, render_template, request
import joblib
import os
import numpy as np
import pickle

app= Flask(__name__)

@app.route("/")
def index():
    return render_template("home.html")

@app.route("/result",methods=['POST','GET'])
def result():
    satisfaction_level=float(request.form['satisfaction'])
    number_project=int(request.form['numberOfProjects'])
    average_montly_hours=int(request.form['avgMonthlyHours'])
    time_spend_company = int(request.form['timeSpendCompany'])
    promotion_last_5years = int(request.form['promotionLast5Years'])
    salary = str(request.form['salary'])

    if salary == 'Low':
        salary_high = 0
        salary_low = 1
        salary_medium = 0

    elif salary == 'Medium':
        salary_high = 0
        salary_low = 0
        salary_medium = 1

    else:
        salary_high = 1
        salary_low = 0
        salary_medium = 0


    x=np.array([satisfaction_level,number_project,average_montly_hours,time_spend_company,promotion_last_5years,
                salary_high,salary_low,salary_medium]).reshape(1,-1)


    model = pickle.load(open('empatr_rf_model.pkl', 'rb'))

    Y_pred=model.predict(x)

    # for No Stroke Risk
    if Y_pred==0:
        return render_template('empretain.html')
    else:
        return render_template('empleft.html')

if __name__=="__main__":
    app.run(debug=True,port=7385)