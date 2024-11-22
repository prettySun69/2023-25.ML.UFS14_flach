import logging
import json
import glob
import sys
from os import environ
from flask import Flask
from keras import models
import numpy as np

logging.debug('Init a Flask app')
app = Flask(__name__)


def doit():
    try:
        iris = load_iris()
        X, y = iris.data, iris.target
        
        X = X[0]
        y = y[0]

        model_dir = environ['SM_MODEL_DIR']
        print(f"######## La model dir è: {model_dir}")
        
        model = models.load_model(f"{model_dir}/my_model.keras")
        # model = models.load_model("my_model.keras")
        
        predict_input1 = np.array([
           
            X
        ])
        predict_result1 = model.predict(predict_input1)

        
        return json.dumps({
            "predict_result": predict_result1.tolist(),
            "expected result": y
        })
    
    except Exception as e:
        return str(e)
    
@app.route('/ping')
def ping():
    logging.debug('Hello from route /ping')

    return doit()
