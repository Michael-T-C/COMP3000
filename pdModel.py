#Numpy used for working with arrays
import numpy as np
#Pandas, an analysis tool with multitude of functions
import pandas as pd

import pickle
import requests
import json


#Support Vector Machine model, gives best results for classification
from sklearn import svm
#Scalar function 
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Load data from csv file into pandas data structure
parkinsons_data = pd.read_csv("data.csv")

#Output head (first 5 rows of data file)
parkinsons_data.head()
# .shape to display datasey array dimensions
parkinsons_data.shape
parkinsons_data.describe()
parkinsons_data.groupby('status').mean()

X = parkinsons_data.drop(columns=['name','status'], axis = 1)
Y = parkinsons_data['status']
print(X)
print(Y)


X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)


scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
print(X_train)


parkinsonsmodel = svm.SVC(kernel='linear')
parkinsonsmodel.fit(X_train, Y_train)


X_train_prediction = parkinsonsmodel.predict(X_train)
trainingdata_accuracy = accuracy_score(Y_train, X_train_prediction)

X_test_prediction = parkinsonsmodel.predict(X_test)
testdata_accuracy = accuracy_score(Y_test, X_test_prediction)

def accuracy():
    #print('Accuracy: ', trainingdata_accuracy)
    #print('Accuracy: ', testdata_accuracy)
    return testdata_accuracy


def process(data_):
    input_parkinsonsdata=data_
    input_parkinsonsdata_as_numpy_array = np.asarray(input_parkinsonsdata)
    input_parkinsonsdata_reshaped = input_parkinsonsdata_as_numpy_array.reshape(1,-1)
    data_standardised = scaler.transform(input_parkinsonsdata_reshaped)
    prediction = parkinsonsmodel.predict(data_standardised)
    return prediction

pickle.dump(parkinsonsmodel, open('pdModel.pkl', 'wb'))
pdModel = pickle.load(open('pdModel.pkl','rb'))






#prediction = pdModel.predict([[153.84800,165.73800,65.78200,0.00840,0.00005,0.00428,0.00450,0.01285,0.03810,0.32800,0.01667,0.02383,0.04055,0.05000,0.03871,17.53600,0.660125,0.704087,-4.095442,0.262564,2.739710,0.365391]])
#print("Model has predicted: ", prediction)

#if(prediction[0]==0):
    #print("No Parkinsons")
#else:
    #print("Parkinsons")
    #print("Prediction Accuracy: ", testdata_accuracy)