from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Încarcă modelul
model = joblib.load('mlb.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  # se presupune că datele sunt trimise ca JSON
        
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
