# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np


app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    int_feature = [int(x) for x in request.form.values()]
    prediction = model.predict([np.array(int_feature)])
    prediction = round(prediction[0],2)
    return render_template('index.html',
                           prediction_text = "the salary is $ {}".format(prediction))

if __name__ == "__main__":
    app.run(debug = True)


                           