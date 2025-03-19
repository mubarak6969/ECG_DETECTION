import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, LSTM, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns

# Load preprocessed data
X_train = np.load('X_train.npy')
X_val = np.load('X_val.npy')
y_train = np.load('y_train.npy')
y_val = np.load('y_val.npy')

# Reshape for CNN (samples, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print("Class weights:", class_weight_dict)

# Function to build and train a model
def build_and_train_model(model_type='cnn', epochs=100, batch_size=32):
    if model_type == 'cnn':
        model = Sequential([
            Conv1D(64, kernel_size=5, activation='relu', input_shape=(200, 1)),
            MaxPooling1D(pool_size=2),
            Conv1D(128, kernel_size=5, activation='relu'),
            MaxPooling1D(pool_size=2),
            Conv1D(256, kernel_size=5, activation='relu'),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
    elif model_type == 'rcnn':
        model = Sequential([
            Conv1D(64, kernel_size=5, activation='relu', input_shape=(200, 1)),
            MaxPooling1D(pool_size=2),
            BatchNormalization(),
            Conv1D(128, kernel_size=5, activation='relu'),
            MaxPooling1D(pool_size=2),
            BatchNormalization(),
            Conv1D(256, kernel_size=5, activation='relu'),
            MaxPooling1D(pool_size=2),
            LSTM(256, return_sequences=True),  # Increased units, keep sequences
            LSTM(128, return_sequences=False),  # Second LSTM layer
            Dense(256, activation='relu'),
            Dropout(0.4),  # Reduced from 0.5
            Dense(1, activation='sigmoid')
        ])
    else:
        raise ValueError("Model type must be 'cnn' or 'rcnn'")

    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    history = model.fit(X_train, y_train, 
                        epochs=epochs, 
                        batch_size=batch_size, 
                        validation_data=(X_val, y_val),
                        class_weight=class_weight_dict,
                        callbacks=[early_stopping, reduce_lr],
                        verbose=1)

    return model, history

# Function to plot training history
def plot_history(history, model_type):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'{model_type.upper()} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{model_type.upper()} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{model_type}_training_plot.png')
    plt.close()

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, model_type):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Abnormal'], yticklabels=['Normal', 'Abnormal'])
    plt.title(f'{model_type.upper()} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'{model_type}_confusion_matrix.png')
    plt.close()

# Train and evaluate CNN
print("Training Baseline CNN...")
cnn_model, cnn_history = build_and_train_model('cnn', epochs=100, batch_size=32)
cnn_model.save('cnn_model.h5')
plot_history(cnn_history, 'cnn')

cnn_pred = (cnn_model.predict(X_val) > 0.5).astype(int)
print("\nCNN Classification Report:")
print(classification_report(y_val, cnn_pred))
print("CNN Confusion Matrix:")
print(confusion_matrix(y_val, cnn_pred))
plot_confusion_matrix(y_val, cnn_pred, 'cnn')

# Train and evaluate RCNN
print("\nTraining RCNN...")
rcnn_model, rcnn_history = build_and_train_model('rcnn', epochs=100, batch_size=32)
rcnn_model.save('rcnn_model.h5')
plot_history(rcnn_history, 'rcnn')

rcnn_pred = (rcnn_model.predict(X_val) > 0.5).astype(int)
print("\nRCNN Classification Report:")
print(classification_report(y_val, rcnn_pred))
print("RCNN Confusion Matrix:")
print(confusion_matrix(y_val, rcnn_pred))
plot_confusion_matrix(y_val, rcnn_pred, 'rcnn')