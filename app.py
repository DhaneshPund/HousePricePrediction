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

@app.route('/test', methods=['POST'])
def test():
    clf = joblib.load('filename.pkl')
    request_data = request.get_json()
    a = request_data['Avg. Area Income']
    b = request_data['Avg. Area House Age']
    c = request_data['Avg. Area Number of Rooms']
    d = request_data['Avg. Area Number of Bedrooms']
    e = request_data['Area Population']
    l = [a, b, c, d, e]
    narr = np.array(l)
    narr = narr.reshape(1, 5)
    df_test = pd.DataFrame(narr, columns=['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms','Avg. Area Number of Bedrooms', 'Area Population'])
    ypred = clf.predict(df_test)
    result = ypred.tolist() #as ndarray object cannot be converted directly to json
    return jsonify({'Price': result})

app.run(port=5000)
