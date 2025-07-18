import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold, cross_validate
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
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

class KerasModelWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model=None):
        self.model = model

    def fit(self, X, y):
        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("Keras model is not defined!")
        y_pred_proba = self.model.predict(X, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        return y_pred

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Keras model is not defined!")
        y_pred_proba = self.model.predict(X, verbose=0)
        return y_pred_proba

    def get_params(self, deep=True):
        return {"model": self.model}

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

def cross_validate_model(global_model, X, y, scaler, X_test):
    print("\nStep 5: 5-Fold cross-validation (time series, 5 classes)")
    print(f"Shape of data X: {X.shape}, y: {y.shape}")

    try:
        X_flat = X.reshape(X.shape[0], -1)
        X_scaled = scaler.transform(X_flat).reshape(X.shape)
        print(f"Shape of scaled data X_scaled: {X_scaled.shape}")
    except Exception as e:
        print(f"Error in scaling data: {e}")
        raise

    wrapped_model = KerasModelWrapper(model=global_model)
    scoring = {'accuracy': 'accuracy', 'f1_macro': 'f1_macro'}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    print("\nClass distribution in folds:")
    for i, (_, test_index) in enumerate(kf.split(X_scaled)):
        y_test_fold = y[test_index]
        print(f"Fold {i+1}: {np.bincount(y_test_fold)}")

    try:
        scores = cross_validate(
            estimator=wrapped_model,
            X=X_scaled,
            y=y,
            cv=kf,
            scoring=scoring,
            return_train_score=False,
            n_jobs=1,
            verbose=0
        )

        print("\nResults for each fold:")
        for i in range(5):
            print(f"Fold {i+1}:")
            print(f"  Accuracy: {scores['test_accuracy'][i]:.4f}")
            print(f"  F1-Score (macro): {scores['test_f1_macro'][i]:.4f}")

        print(f"\nAverage results (5-Fold):")
        print(f"  Accuracy: {scores['test_accuracy'].mean():.4f} ± {scores['test_accuracy'].std():.4f}")
        print(f"  F1-Score (macro): {scores['test_f1_macro'].mean():.4f} ± {scores['test_f1_macro'].std():.4f}")

        return scores
    except Exception as e:
        print(f"Error in executing cross_validate: {e}")
        raise

if __name__ == "__main__":
    try:
        X = np.load(os.path.join(BASE_PATH, 'X_5class_timeseries168-1.npy'))
        y = np.load(os.path.join(BASE_PATH, 'y_5class_timeseries168-1.npy'))
        X_test = np.load(os.path.join(BASE_PATH, 'X_test_5class_timeseries168-1.npy'))
        scaler = StandardScaler().fit(X.reshape(X.shape[0], -1))
        global_model = create_local_model()
        global_model.load_weights(os.path.join(BASE_PATH, 'global_model_5class_timeseries168-1.weights.h5'))
        cross_validate_model(global_model, X, y, scaler, X_test)
    except FileNotFoundError as e:
        print(f"Error in loading files: {e}")
        raise