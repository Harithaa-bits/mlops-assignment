from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Expecting JSON data
    input_data = np.array(data['input']).reshape(1, -1)  # Reshaping input for prediction
    prediction = model.predict(input_data)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == "__main__":
    # Run the app on port 5000 (can be any port, just make sure it's open in the security group)
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000
    )
