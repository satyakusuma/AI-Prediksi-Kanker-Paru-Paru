from flask import Flask, render_template, request
from joblib import load
import numpy as np
import os

app = Flask(__name__, template_folder='templates')

# Pastikan path model benar
model_path = os.path.join(os.path.dirname(__file__), 'knn_model.joblib')
try:
    model = load(model_path)
except FileNotFoundError:
    print("Error: knn_model.joblib not found!")
    raise

features_order = [
    'yellow_fingers', 'anxiety', 'peer_pressure', 'chronic_disease',
    'fatigue', 'allergy', 'wheezing', 'alcohol_consuming',
    'coughing', 'swallowing_difficulty', 'chest_pain'
]

def map_output(prediction):
    return "YA" if prediction[0] == 1 else "TIDAK"

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', features=features_order, error=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [int(request.form[feature]) for feature in features_order]
        if not all(x in [1, 2] for x in features):
            return render_template('index.html', features=features_order, error="Semua jawaban harus 'Tidak' (1) atau 'Ya' (2)!")
        input_data = np.array(features).reshape(1, -1)
        prediction = model.predict(input_data)
        result = map_output(prediction)
        return render_template('result.html', prediction=result)
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return render_template('index.html', features=features_order, error=f"Terjadi kesalahan: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)