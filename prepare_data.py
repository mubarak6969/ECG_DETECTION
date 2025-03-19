import pandas as pd
import ast
import wfdb
import numpy as np
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
import os

# Load metadata
data_path = r"data\ptbxl\ptbxl_database.csv"
df = pd.read_csv(data_path)

# Function to classify ECGs as normal (0) or abnormal (1)
def classify_ecg(scp_str):
    try:
        scp_dict = ast.literal_eval(scp_str)
        if 'NORM' in scp_dict and len(scp_dict) == 1:
            return 0
        return 1
    except (ValueError, SyntaxError):
        return 1

# Apply classification
df['label'] = df['scp_codes'].apply(classify_ecg)
print("Full dataset label distribution:")
print(df['label'].value_counts())

# Balance the dataset
normal_df = df[df['label'] == 0]  # All 190 normal samples
abnormal_df = df[df['label'] == 1].sample(n=len(normal_df) * 2, random_state=42)  # 2x normal
balanced_df = pd.concat([normal_df, abnormal_df]).sample(frac=1, random_state=42)
print("Balanced dataset label distribution:")
print(balanced_df['label'].value_counts())

# Function to detect R-peaks and segment ECG signals
def get_rpeak_segments(filename, window_size=200):
    try:
        record = wfdb.rdrecord(filename, sampfrom=0, physical=False)
        signal = record.d_signal[:, 1]  # Lead II
        peaks, _ = find_peaks(signal, height=np.max(signal)*0.5, distance=50)
        segments = []
        half_window = window_size // 2
        for peak in peaks:
            start = max(0, peak - half_window)
            end = min(len(signal), peak + half_window)
            if end - start == window_size:
                segments.append(signal[start:end])
        return np.array(segments)
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return np.array([])

# Process balanced records
data_dir = r"data\ptbxl"
all_segments = []
all_labels = []

for idx, row in balanced_df.iterrows():
    filename = os.path.join(data_dir, row['filename_hr'])
    segments = get_rpeak_segments(filename)
    if segments.size > 0:
        all_segments.append(segments)
        all_labels.extend([row['label']] * len(segments))
    print(f"Processed ECG {row['ecg_id']}: {segments.shape} segments")

# Convert to numpy arrays
if all_segments:
    X = np.concatenate(all_segments, axis=0)
    y = np.array(all_labels)
    print(f"Total segments: {X.shape}, Labels: {y.shape}")
    print("Segment label distribution:", np.unique(y, return_counts=True))
    
    # Save full dataset
    np.save('segments.npy', X)
    np.save('labels.npy', y)
    print("Segments and labels saved to 'segments.npy' and 'labels.npy'")
else:
    print("No segments extracted. Check R-peak detection or dataset.")

# Split data with stratification
if all_segments:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")
    print("y_train distribution:", np.unique(y_train, return_counts=True))
    print("y_val distribution:", np.unique(y_val, return_counts=True))
    
    # Save split data
    np.save('X_train.npy', X_train)
    np.save('X_val.npy', X_val)
    np.save('y_train.npy', y_train)
    np.save('y_val.npy', y_val)
    print("Training and validation sets saved as .npy files")
    
    # Save metadata
    train_df, val_df = train_test_split(balanced_df, test_size=0.2, stratify=balanced_df['label'], random_state=42)
    train_df.to_csv('train_data.csv', index=False)
    val_df.to_csv('val_data.csv', index=False)
    print(f"Training metadata: {len(train_df)} samples, Validation metadata: {len(val_df)} samples")
else:
    print("Skipping split due to no segments.")