# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 14:16:23 2021

@author: Administrator
"""

import pickle
from flask import Flask,render_template,request
import numpy as np
app=Flask(__name__)
model=pickle.load(open("model.pkl","rb"))
 
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict",methods=["POST"])
def predict():
     '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values("experience","test_score(out of 10)","interview_score(out of 10)")]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))


if __name__=="__main__":
    app.run(debug=True)