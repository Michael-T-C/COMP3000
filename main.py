import imp
from flask import Flask, render_template, redirect, request, jsonify
import numpy as np
import pickle
import csv


from requests import post

from pdModel import process as Process
from pdModel import accuracy as Accuracy
app = Flask(__name__)
pdModel = pickle.load(open('pdModel.pkl','rb'))


@app.route('/home', methods=['POST', 'GET']) #Connects to Flask URL (127.0.0.1:5000/home)
def home():
    if request.method == 'POST': #If live html instance asks for POST method (i.e data has been entered and submitted)
        valList = [] #Empty list that will store inputs
        for val in request.form:
            valList.append(request.form[val]) #Adds each input onto list
        print(valList)
        return render_template('base.html', data1=valList[0], data2=valList[1]) #Two data inputs are sent through to html page as variables.
    else:
        return render_template('base.html') #GET methods

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        rawFile = request.form['csvfile']
        data = []
        with open(rawFile) as file:
            csvFile = csv.reader(file)
            
            for row in csvFile:
                print("row before: ",row)
                del row[0]
                del row[16]
                data.append(tuple(row))
                print("row after: ",row)

                prediction = Process(data[0])
                accuracy = Accuracy()*100
                print('PRED:', prediction)
            

        if int(prediction) == 0:
            return render_template('predict.html', prediction='Patient does not have parkinsons', accuracy=int(accuracy))
        elif int(prediction) == 1:
            return render_template('predict.html', prediction='Patient has parkinsons', accuracy=int(accuracy))
            

        


if __name__ == "__main__":
    app.run(debug=True)

