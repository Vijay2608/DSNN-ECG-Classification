import numpy as np
import wfdb
import os

LABELS = {
    'N': 0,  # Normal
    'L': 1,  # Left Bundle Branch Block (LBBB)
    'R': 2,  # Right Bundle Branch Block (RBBB)
    'A': 3,  # Atrial Premature Contraction (APC)
    'V': 4,  # Premature Ventricular Contraction (PVC)
    '/': 5   # Paced beat
}

def extract_heartbeats(data_path='mitbih_data'):
    X, y = [], []
    for filename in os.listdir(data_path):
        if filename.endswith('.dat'):
            record_name = filename[:-4]
            record_path = os.path.join(data_path, record_name)
            record = wfdb.rdrecord(record_path)
            annotation = wfdb.rdann(record_path, 'atr')
            for i, label in enumerate(annotation.symbol):
                if label in LABELS:
                    index = annotation.sample[i]
                    if index - 90 >= 0 and index + 91 < len(record.p_signal):
                        beat = record.p_signal[index - 90:index + 91, 0]
                        beat = (beat - np.mean(beat)) / (np.std(beat) + 1e-8)
                        X.append(beat.reshape(1, -1))
                        y.append(LABELS[label])
    return np.array(X), np.array(y)

def create_shifted(X):
    shifted = []
    for x in X:
        shift = np.random.randint(-3, 4)
        shifted.append(np.roll(x, shift, axis=1))
    return np.array(shifted)
