import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os
print(os.getcwd())

app_flask = Flask(__name__)  # name is not standard
model = pickle.load(open("flask_test_model.pkl", 'rb'))


@app_flask.route('/')
def home():
    return render_template('index.html')


@app_flask.route('/predict', methods=['POST'])
def predict():
    """
    for rendering results on html GUI

    """
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))


if __name__ == "__main__":
    app_flask.run(debug=True)
