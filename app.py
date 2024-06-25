from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app, support_credentials=True)

# Load the model
model_path = 'mlb.pkl'
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    raise FileNotFoundError(f"Model file {model_path} not found.")

# Define a route for the root URL
@app.route('/')
def home():
    return "Welcome to the prediction API. Use the /predict endpoint to get predictions."

# Define a route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Assume data is sent as JSON
        data = request.json  
        
        # Check if data is a dict or a list
        if not isinstance(data, (list, dict)):
            raise ValueError("Input data should be a list or a dictionary.")

        # Make the prediction
        prediction = model.predict([data])
        
        # Return the prediction result
        return jsonify({'prediction': prediction.tolist()}), 200
    except Exception as e:
        # In case of an error, return the error message
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Run the Flask application in development mode
    app.run(debug=True)
