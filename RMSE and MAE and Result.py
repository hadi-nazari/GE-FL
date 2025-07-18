import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from sklearn.model_selection import KFold
import seaborn as sns
import matplotlib.pyplot as plt
import os

BASE_PATH = "C:/Users/Work-User/Desktop/model/"

# Mapping classes to PM2.5 values (µg/m³) based on dataset averages
CLASS_TO_PM25 = {
    0: 7.68,   # L0
    1: 22.5,   # L1
    2: 45.17,  # L2
    3: 94.12,  # L3
    4: 190.0   # L4 (testing with 207.0 is also suggested)
}

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

def evaluate_global_model(global_model, X_test, y_test):
    print("\nStep 4: Evaluating global model (time series, 5 classes)")
    try:
        # Prediction
        y_pred_proba = global_model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Calculate accuracy and model size
        accuracy = accuracy_score(y_test, y_pred)
        weights = get_weights(global_model)
        size = np.count_nonzero(weights) / len(weights)
        print(f"Global model accuracy: {accuracy:.4f}")
        print(f"Model size (ratio of non-zero weights): {size:.4f}")

        # Convert classes to PM2.5 values
        y_test_pm25 = np.array([CLASS_TO_PM25[y] for y in y_test])
        y_pred_pm25 = np.array([CLASS_TO_PM25[y] for y in y_pred])

        # Calculate RMSE, MAPE, and MAE with cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        rmse_scores = []
        mape_scores = []
        mae_scores = []
        for train_index, test_index in kf.split(X_test):
            X_test_fold, y_test_fold = X_test[test_index], y_test[test_index]
            y_pred_proba_fold = global_model.predict(X_test_fold, verbose=0)
            y_pred_fold = np.argmax(y_pred_proba_fold, axis=1)
            y_test_fold_pm25 = np.array([CLASS_TO_PM25[y] for y in y_test_fold])
            y_pred_fold_pm25 = np.array([CLASS_TO_PM25[y] for y in y_pred_fold])
            rmse = np.sqrt(mean_squared_error(y_test_fold_pm25, y_pred_fold_pm25))
            mape = mean_absolute_percentage_error(y_test_fold_pm25, y_pred_fold_pm25) * 100  # Percentage
            mae = mean_absolute_error(y_test_fold_pm25, y_pred_fold_pm25)  # µg/m³
            rmse_scores.append(rmse)
            mape_scores.append(mape)
            mae_scores.append(mae)

        mean_rmse = np.mean(rmse_scores)
        std_rmse = np.std(rmse_scores)
        mean_mape = np.mean(mape_scores)
        std_mape = np.std(mape_scores)
        mean_mae = np.mean(mae_scores)
        std_mae = np.std(mae_scores)
        print(f"Mean RMSE (cross-validation): {mean_rmse:.2f} µg/m³ ± {std_rmse:.2f}")
        print(f"Mean MAPE (cross-validation): {mean_mape:.2f}% ± {std_mape:.2f}")
        print(f"Mean MAE (cross-validation): {mean_mae:.2f} µg/m³ ± {std_mae:.2f}")

        # Classification report
        print("\nClassification report:")
        print(classification_report(y_test, y_pred, target_names=['L0', 'L1', 'L2', 'L3', 'L4']))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['L0', 'L1', 'L2', 'L3', 'L4'],
                    yticklabels=['L0', 'L1', 'L2', 'L3', 'L4'])
        plt.title('Confusion Matrix (5 Classes)')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(BASE_PATH, 'confusion_matrix_5class_timeseries168-1.png'))
        plt.show()

        # Save metrics
        np.save(os.path.join(BASE_PATH, 'rmse_scores_5class_timeseries168-1.npy'), rmse_scores)
        np.save(os.path.join(BASE_PATH, 'mape_scores_5class_timeseries168-1.npy'), mape_scores)
        np.save(os.path.join(BASE_PATH, 'mae_scores_5class_timeseries168-1.npy'), mae_scores)

    except Exception as e:
        print(f"Error in evaluating global model: {e}")
        raise

if __name__ == "__main__":
    try:
        X_test = np.load(os.path.join(BASE_PATH, 'X_test_5class_timeseries168-1.npy'))
        y_test = np.load(os.path.join(BASE_PATH, 'y_test_5class_timeseries168-1.npy'))
        global_model = create_local_model()
        global_model.load_weights(os.path.join(BASE_PATH, 'global_model_5class_timeseries168-12.weights.h5'))
        evaluate_global_model(global_model, X_test, y_test)
    except FileNotFoundError as e:
        print(f"Error in loading files: {e}")
        raise