# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
step 1:start the program

step 2: Import the standard Libraries.

stap 3:Set variables for assigning dataset values.

step 4:Import linear regression from sklearn.

step 5:Assign the points for representing in the graph.

step 6:Predict the regression for marks by using the representation of the graph.

step 7: Compare the graphs and hence we obtained the linear regression for the given datas.

step 8:End the program
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Mithunraj M
RegisterNumber: 212222040100 
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("C:/Users/SEC/Downloads/Placement_Data.csv")
dataset

dataset = dataset.drop('sl_no' ,axis = 1)
dataset = dataset.drop('salary' ,axis = 1)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
dataset["gender"]=dataset["gender"].astype('category' )
dataset["ssc_b"]=dataset["ssc_b"].astype('category' )
dataset["hsc_b"]=dataset["hsc_b"].astype('category' )
dataset["hsc_s"]=dataset["hsc_s"].astype('category' )
dataset["degree_t"]=dataset["degree_t"].astype('category' )
dataset["workex"]=dataset["workex"].astype('category' )
dataset["specialisation"]=dataset["specialisation"].astype('category' )
dataset["status"]=dataset["status"].astype('category' )
dataset.dtypes

dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset

X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, -1].values
Y

theta=np.random.randn(X.shape[1])
y=Y

def sigmoid(z):
    return 1/(1+np.exp(-z))

def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return  -np.sum(y*np.log(h)+(1-y)*log(1-h))

def gradient_descent(theta,X,y,alpha,num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h=sigmoid(X.dot(theta))
        gradient=X.T.dot(h-y)/m
        theta=alpha*gradient
    return theta

theta=gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)

def predict(theta,X):
    h=sigmoid(X.dot(theta))
    y_pred=np.where(h>=0.5,1,0)
    return y_pred
y_pred=predict(theta,X)

accuracy=np.mean(y_pred.flatten()==y)
print('Accuracy:' ,accuracy)

print(y_pred)

print(Y)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)

xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```

## Output:

![exp 5 ML output accuracy](https://github.com/user-attachments/assets/13d7d14d-283e-402f-83d4-eb6d50938282)

![exp 5 ml output](https://github.com/user-attachments/assets/c3177201-193f-4376-9dfe-d603f0dc04b4)




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
