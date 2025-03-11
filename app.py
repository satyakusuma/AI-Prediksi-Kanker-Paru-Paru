from flask import Flask, render_template, request
from joblib import load
import numpy as np

app = Flask(__name__)

# Muat model
model = load('knn_model.joblib')

# Daftar fitur yang digunakan dalam model
features_order = [
    'yellow_fingers', 'anxiety', 'peer_pressure', 'chronic_disease',
    'fatigue', 'allergy', 'wheezing', 'alcohol_consuming',
    'coughing', 'swallowing_difficulty', 'chest_pain'
]

# Fungsi untuk memetakan output model
def map_output(prediction):
    return "YA" if prediction[0] == 1 else "TIDAK"

# Halaman utama
@app.route('/')
def home():
    return render_template('index.html', features=features_order, error=None)

# Halaman prediksi
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data dari formulir
        features = [int(request.form[feature]) for feature in features_order]
        # Validasi input (harus 1 atau 2)
        if not all(x in [1, 2] for x in features):
            return render_template('index.html', features=features_order, error="Semua jawaban harus 'Tidak' (1) atau 'Ya' (2)!")
        # Konversi ke array numpy
        input_data = np.array(features).reshape(1, -1)
        # Prediksi
        prediction = model.predict(input_data)
        result = map_output(prediction)
        return render_template('result.html', prediction=result)
    except Exception as e:
        return render_template('index.html', features=features_order, error=f"Terjadi kesalahan: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)