import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os

BASE_PATH = "C:/Users/Work-User/Desktop/model/"

def create_local_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(168, 4)),
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.01)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def get_weights(model):
    weights = model.get_weights()
    flattened = [w.flatten() for w in weights]
    try:
        concatenated = np.concatenate(flattened)
        return concatenated
    except ValueError as e:
        print(f"Error in concatenating weights: {e}")
        raise

def train_local_models(X_nodes, y_nodes, X_test, y_test):
    print("\nStep 2: Local training on nodes (time series, 5 classes)")
    local_models = []
    for i, (X_n, y_n) in enumerate(zip(X_nodes, y_nodes)):
        print(f"\nNode {i+1}:")
        if len(X_n) < 2:
            print("Not enough data")
            continue
        model = create_local_model()
        X_train, X_test_local, y_train, y_test_local = train_test_split(
            X_n, y_n, test_size=0.2, random_state=42, stratify=y_n
        )
        print(f"Training data size: {len(X_train)}, local test data: {len(X_test_local)}")
        print(f"Class distribution in training data of node {i+1}: {np.bincount(y_train)}")

        # Adjusted class weights
        class_weights = {0: 3.0, 1: 2.0, 2: 1.0, 3: 1.0, 4: 1.0}  # More balanced
        model.fit(X_train, y_train, epochs=10, batch_size=128, verbose=0, class_weight=class_weights)

        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"Local accuracy (on central test): {accuracy:.4f}, loss: {loss:.4f}")
        local_models.append(model)

    if not local_models:
        raise ValueError("No local models were trained!")

    for i, model in enumerate(local_models):
        model.save_weights(os.path.join(BASE_PATH, f'local_model_{i}_5class_timeseries168-1.weights.h5'))

    return local_models

if __name__ == "__main__":
    tf.random.set_seed(42)
    try:
        X_nodes = [np.load(os.path.join(BASE_PATH, f'X_node_{i}_5class_timeseries168-1.npy')) for i in range(5)]
        y_nodes = [np.load(os.path.join(BASE_PATH, f'y_node_{i}_5class_timeseries168-1.npy')) for i in range(5)]
        X_test = np.load(os.path.join(BASE_PATH, 'X_test_5class_timeseries168-1.npy'))
        y_test = np.load(os.path.join(BASE_PATH, 'y_test_5class_timeseries168-1.npy'))
    except FileNotFoundError as e:
        print(f"Error in loading data: {e}")
        raise
    local_models = train_local_models(X_nodes, y_nodes, X_test, y_test)