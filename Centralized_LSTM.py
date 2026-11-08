import os
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error
)

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

BASE_PATH = "/content/drive/My Drive/beijing_data/"

X_TRAIN_PATH = os.path.join(
    BASE_PATH,
    "X_centralized_5class_timeseries168-1.npy"
)

Y_TRAIN_PATH = os.path.join(
    BASE_PATH,
    "y_centralized_5class_timeseries168-1.npy"
)

X_TEST_PATH = os.path.join(
    BASE_PATH,
    "X_test_centralized_5class_timeseries168-1.npy"
)

Y_TEST_PATH = os.path.join(
    BASE_PATH,
    "y_test_centralized_5class_timeseries168-1.npy"
)

# ============================================================
# 1. Load data
# ============================================================

X = np.load(X_TRAIN_PATH)
y = np.load(Y_TRAIN_PATH)

X_test = np.load(X_TEST_PATH)
y_test = np.load(Y_TEST_PATH)

print("Centralized Train:", X.shape)
print("Centralized Test :", X_test.shape)

# ============================================================
# 2. Train / Validation split
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training   :", X_train.shape)
print("Validation :", X_val.shape)

# ============================================================
# 3. Model
# ============================================================

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(168, 4)
    ),

    tf.keras.layers.LSTM(
        64,
        return_sequences=False
    ),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(
        16,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l1(0.01)
    ),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(
        5,
        activation="softmax"
    )
])

# ============================================================
# 4. Compile
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ============================================================
# 5. Class weights
# ============================================================

class_weights = {
    0: 3.0,
    1: 2.0,
    2: 1.0,
    3: 1.0,
    4: 1.0
}

# ============================================================
# 6. Training
# ============================================================

history = model.fit(
    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=10,
    batch_size=128,

    class_weight=class_weights,

    verbose=1
)

# ============================================================
# 7. Test evaluation
# ============================================================

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

# ============================================================
# 8. Predictions
# ============================================================

y_prob = model.predict(
    X_test,
    verbose=0
)

y_pred = np.argmax(
    y_prob,
    axis=1
)

# ============================================================
# 9. Metrics
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro"
)

weighted_f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

f1_per_class = f1_score(
    y_test,
    y_pred,
    average=None,
    labels=[0, 1, 2, 3, 4]
)

custom_f1 = np.average(
    f1_per_class,
    weights=[3, 2, 1, 1, 1]
)

# ============================================================
# 10. RMSE / MAE
# ============================================================

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

mae = mean_absolute_error(
    y_test,
    y_pred
)

# ============================================================
# 11. Results
# ============================================================

print("\n" + "=" * 60)
print("CENTRALIZED LSTM RESULTS")
print("=" * 60)

print(f"Test Loss          : {test_loss:.4f}")
print(f"Accuracy           : {accuracy:.4f}")
print(f"Macro F1           : {macro_f1:.4f}")
print(f"Weighted F1        : {weighted_f1:.4f}")
print(f"Custom Weighted F1 : {custom_f1:.4f}")
print(f"RMSE               : {rmse:.4f}")
print(f"MAE                : {mae:.4f}")

# ============================================================
# 12. Classification report
# ============================================================

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        digits=4
    )
)

# ============================================================
# 13. Confusion Matrix
# ============================================================

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

# ============================================================
# 14. Parameter count
# ============================================================

total_params = model.count_params()

print("\nTotal Parameters:", total_params)

# ============================================================
# 15. Save model
# ============================================================

MODEL_PATH = os.path.join(
    BASE_PATH,
    "Centralized_LSTM_5class_168_24.keras"
)

model.save(MODEL_PATH)

print("\nModel saved to:")
print(MODEL_PATH)
