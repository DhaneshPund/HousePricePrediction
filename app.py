from flask import Flask, jsonify, request

app = Flask(__name__)

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib


@app.route('/train')
def train():
    df_train = pd.read_csv('USA_Housing.csv')
    df_train = df_train.iloc[:, 0:6] #dropping Address column
    X = df_train.iloc[:, 0:5] #vertical split
    y = df_train['Price'] #vertical split
    classifier = LinearRegression()
    classifier.fit(X, y)
    joblib.dump(classifier, 'filename.pkl')

    return 'Model has been Trained'

app.run(port=5000)
