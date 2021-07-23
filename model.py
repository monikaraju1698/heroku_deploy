# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 09:24:53 2021

@author: Administrator
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
df=pd.read_csv(r"C:\Users\Administrator\Desktop\datasets\hiring.csv")
df.head()
df.tail()
df.isna().sum()
df["experience"]=df["experience"].fillna("zero")
df["test_score(out of 10)"]=df["test_score(out of 10)"].fillna(0)
from sklearn.preprocessing import LabelEncoder
trans=LabelEncoder()
df["experience"]=trans.fit_transform(df["experience"])
x=df.iloc[:,:-1]
y=df.iloc[:,-1]
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x,y)
pickle.dump(model,open("model.pkl","wb"))
model=pickle.load(open("model.pkl","rb"))
print(model.predict([[2,6,6]]))