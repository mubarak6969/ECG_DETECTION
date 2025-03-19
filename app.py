import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.saving import load_model
import wfdb
from scipy.signal import find_peaks
import os
import shutil

app = Flask(__name__)

# Load RCNN model
model = load_model('rcnn_model.h5', compile=False)

# Preprocessing function
def preprocess_ecg(file_path, window_size=200):
    try:
        print(f"Processing file: {file_path}")
        if os.path.exists(file_path + '.hea'):
            print("Using .hea file with rdrecord")
            record = wfdb.rdrecord(file_path, physical=False)
            signal = record.d_signal[:, 1]  # Lead II
        else:
            print("No .hea file found, attempting to read .dat directly")
            with open(file_path + '.dat', 'rb') as f:
                raw_data = np.fromfile(f, dtype=np.int16)
            num_samples = len(raw_data) // 12
            signal_data = raw_data.reshape(num_samples, 12)
            signal = signal_data[:, 1]  # Lead II
        peaks, _ = find_peaks(signal, height=np.max(signal)*0.3, distance=30)
        print(f"Found {len(peaks)} R-peaks")
        segments = []
        half_window = window_size // 2
        for peak in peaks:
            start = max(0, peak - half_window)
            end = min(len(signal), peak + half_window)
            if end - start == window_size:
                segments.append(signal[start:end])
        if not segments:
            print("No valid segments extracted")
            return None, None
        X = np.array(segments).reshape(-1, window_size, 1)
        print(f"Segments shape: {X.shape}")
        return X, signal[:1000]  # Return signal snippet for graph
    except Exception as e:
        print(f"Preprocessing error: {str(e)}")
        return str(e), None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        print("No file uploaded")
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    filename = file.filename
    file_path = os.path.join('uploads', filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(file_path)
    print(f"File saved to: {file_path}")

    # Copy .hea file if it exists
    hea_filename = filename.replace('.dat', '.hea')
    hea_source = os.path.join('.', hea_filename)
    hea_dest = os.path.join('uploads', hea_filename)
    if os.path.exists(hea_source):
        shutil.copy(hea_source, hea_dest)
        print(f"Copied .hea to: {hea_dest}")
    else:
        print("No .hea file found in root directory")

    # Preprocess
    result, signal_snippet = preprocess_ecg(file_path[:-4])
    if result is None:
        print("No valid segments extracted")
        return jsonify({'error': 'No valid segments extracted'}), 400
    if isinstance(result, str):
        print(f"Preprocessing failed: {result}")
        return jsonify({'error': result}), 500
    X = result

    # Predict
    try:
        predictions = model.predict(X)
        avg_pred = np.mean(predictions > 0.5)
        label = 1 if avg_pred > 0.5 else 0
        confidence = float(avg_pred) if label else float(1 - avg_pred)
        print(f"Prediction: {label}, Confidence: {confidence}")
        
        prediction = 'abnormal' if label else 'normal'
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

    # Clean up
    os.remove(file_path)
    if os.path.exists(hea_dest):
        os.remove(hea_dest)

    return jsonify({
        'prediction': prediction,
        'signal': signal_snippet.tolist()  # For graph
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)