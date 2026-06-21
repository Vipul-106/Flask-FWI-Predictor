from flask import Flask, render_template, jsonify, request, redirect, url_for
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# Initialize the Flask application
application = Flask(__name__)
app = application

#import ridge regressor and Standard Scaler pickle
ridge_model = pickle.load(open('Models/ridge.pkl', 'rb'))
standar_scaler = pickle.load(open('Models/scaler.pkl', 'rb'))

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['Get', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        Temperature = float(request.form.get("Temperature"))
        RH = float(request.form.get("RH"))
        Ws = float(request.form.get("WS"))
        Rain = float(request.form.get("Rain"))      
        FFMC = float(request.form.get("FFMC"))
        DMC = float(request.form.get("DMC"))
        ISI = float(request.form.get("ISI"))
        Classes = float(request.form.get("Classes"))
        Region = float(request.form.get("Region"))

        input_data = pd.DataFrame([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]], 
                                   columns=['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'ISI', 'Classes', 'Region'])

        
        new_scaled_values = standar_scaler.transform(input_data)
        result = ridge_model.predict(new_scaled_values)

        return render_template('Home.html', results=result[0])

    else:
        return render_template('Home.html')
    

# Run the app if this file is executed directly
if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)