from flask import Flask, request, jsonify
import joblib
import numpy as np


app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Expecting JSON data
    prediction = model.predict(np.array(data['input']).reshape(1, -1))
    return jsonify({'prediction': prediction.tolist()})


if __name__ == "__main__":
    app.run(debug=True)
