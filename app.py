from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Încarcă modelul
model_path = 'mlb.pkl'
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    raise FileNotFoundError(f"Model file {model_path} not found.")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  # se presupune că datele sunt trimise ca JSON
        # Verifică dacă datele sunt un dict sau o listă
        if not isinstance(data, (list, dict)):
            raise ValueError("Input data should be a list or a dictionary.")

        # Realizează predicția
        prediction = model.predict(data)
        
        # Returnează rezultatul predicției
        return jsonify({'prediction': prediction.tolist()}), 200
    except Exception as e:
        # În cazul unei erori, returnează mesajul de eroare
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Rulează aplicația Flask în modul de dezvoltare
    app.run(debug=True)
