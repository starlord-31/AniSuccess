#!/usr/bin/env python
# coding: utf-8

import pickle
import xgboost as xgb
from flask import Flask
from flask import request
from flask import jsonify

with open('model_C=1.0.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


app = Flask(__name__)

@app.route('/predict', methods=['POST'])

def predict():
    new_data_point = request.get_json()

    X_new = dv.transform([new_data_point])
    dnew = xgb.DMatrix(X_new)

    y_pred = model.predict(dnew)
    success_probability = float(y_pred[0])
    success = success_probability >= 0.5

    result = {
        "success_probability": success_probability,
        "success": success
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)