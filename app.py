from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Load the classifier, MultiLabelBinarizer, and vectorizer
classifier_path = 'classifier_model.pkl'
mlb_path = 'mlb.pkl'
vectorizer_path = 'vectorizer.pkl'


if os.path.exists(classifier_path) and os.path.exists(mlb_path) and os.path.exists(vectorizer_path):
    classifier = joblib.load(classifier_path)
    mlb = joblib.load(mlb_path)
    vectorizer = joblib.load(vectorizer_path)
else:
    raise FileNotFoundError(f"Model files not found.")

# Define a route for the root URL
@app.route('/')
def home():
    return "Welcome to the prediction API. Use the /predict endpoint to get predictions."

# Define a route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Log the receipt of a request
        print("Received a prediction request.")
        
        # Assume data is sent as JSON
        data = request.json
        
        # Log the received data
        print(f"Received data: {data}")
        
        # Check if data is a list
        if not isinstance(data, list):
            raise ValueError("Input data should be a list of ingredients.")
        
        # Log before transforming data
        print(f"Transforming data: {data}")
        
        # Transform the input data to the format expected by the classifier
        input_vector = vectorizer.transform([" ".join(data)])
        
        # Log before making a prediction
        print("Making prediction.")
        
        # Make the prediction
        prediction = classifier.predict(input_vector)
        
        # Log the prediction result
        print(f"Prediction result: {prediction}")
        
        # Inverse transform the prediction to get the original labels
        predicted_labels = mlb.inverse_transform(prediction)
        
        # Log the predicted labels
        print(f"Predicted Food Type Labels: {predicted_labels}")
        
        # Return the prediction result
        return jsonify({'prediction': list(predicted_labels)}), 200
    except Exception as e:
        # Log the error
        print(f"Error occurred: {e}")
        
        # In case of an error, return the error message
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Run the Flask application in development mode
    app.run(debug=True)
