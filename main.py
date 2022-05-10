from flask import Flask, render_template, redirect, request


app = Flask(__name__)

@app.route('/home', methods=['POST', 'GET']) #Connects to Flask URL (127.0.0.1:5000/home)
def home():
    if request.method == 'POST': #If live html instance asks for POST method (i.e data has been entered and submitted)
        valList = [] #Empty list that will store inputs
        for val in request.form:
            valList.append(request.form[val]) #Adds each input onto list
        print(valList)
        return render_template('base.html', apple=valList[0], banana=valList[1]) #Two data inputs are sent through to html page as variables.
    else:
        return render_template('base.html') #GET methods



if __name__ == "__main__":
    app.run(debug=True)

