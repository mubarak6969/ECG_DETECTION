import numpy as np
import streamlit as st
from tensorflow.keras.saving import load_model
import wfdb
from scipy.signal import find_peaks
import os
import shutil
import plotly.graph_objects as go

# Load RCNN model
@st.cache_resource
def load_rcnn_model():
    return load_model('rcnn_model.h5', compile=False)

model = load_rcnn_model()

# Preprocessing function
def preprocess_ecg(file_path, window_size=200):
    try:
        if os.path.exists(file_path + '.hea'):
            st.write("Using .hea file with rdrecord")
            record = wfdb.rdrecord(file_path, physical=False)
            signal = record.d_signal[:, 1]  # Lead II
        else:
            st.write("No .hea file found, attempting to read .dat directly")
            with open(file_path + '.dat', 'rb') as f:
                raw_data = np.fromfile(f, dtype=np.int16)
            num_samples = len(raw_data) // 12
            signal_data = raw_data.reshape(num_samples, 12)
            signal = signal_data[:, 1]  # Lead II
        peaks, _ = find_peaks(signal, height=np.max(signal)*0.3, distance=30)
        st.write(f"Found {len(peaks)} R-peaks")
        segments = []
        half_window = window_size // 2
        for peak in peaks:
            start = max(0, peak - half_window)
            end = min(len(signal), peak + half_window)
            if end - start == window_size:
                segments.append(signal[start:end])
        if not segments:
            st.write("No valid segments extracted")
            return None, None
        X = np.array(segments).reshape(-1, window_size, 1)
        st.write(f"Segments shape: {X.shape}")
        return X, signal[:1000]  # Return signal snippet for graph
    except Exception as e:
        st.error(f"Preprocessing error: {str(e)}")
        return None, None

# Streamlit UI
st.title("ECG Classification")
st.write("Upload an ECG file (.dat) to classify it.")

uploaded_file = st.file_uploader("Choose a .dat file", type="dat")

if uploaded_file is not None:
    # Save uploaded file
    file_path = os.path.join('uploads', uploaded_file.name)
    os.makedirs('uploads', exist_ok=True)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    st.write(f"File saved to: {file_path}")

    # Copy .hea file if it exists
    hea_filename = uploaded_file.name.replace('.dat', '.hea')
    hea_source = os.path.join('.', hea_filename)
    hea_dest = os.path.join('uploads', hea_filename)
    if os.path.exists(hea_source):
        shutil.copy(hea_source, hea_dest)
        st.write(f"Copied .hea to: {hea_dest}")
    else:
        st.write("No .hea file found in root directory")

    # Preprocess and predict
    X, signal_snippet = preprocess_ecg(file_path[:-4])
    if X is not None:
        predictions = model.predict(X)
        avg_pred = np.mean(predictions > 0.5)
        label = 1 if avg_pred > 0.5 else 0
        prediction = 'abnormal' if label else 'normal'
        st.success(f"Prediction: **{prediction}**")

        # Plot ECG signal
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(signal_snippet))), y=signal_snippet, mode='lines', name='ECG Signal (Lead II)', line=dict(color='blue')))
        fig.update_layout(title='ECG Signal', xaxis_title='Sample', yaxis_title='Amplitude')
        st.plotly_chart(fig)

    # Clean up
    os.remove(file_path)
    if os.path.exists(hea_dest):
        os.remove(hea_dest)