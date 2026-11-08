# ============================================================
# FedAvg + Pruning
# Beijing PM2.5 5-Class Time-Series Classification
# ============================================================

import os
import gc
import random
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error
)

# ============================================================
# 1. Configuration
# ============================================================

BASE_PATH = "/content/drive/My Drive/beijing_data/"

NUM_NODES = 5
ROUNDS = 1
LOCAL_EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.001

PRUNING_THRESHOLD = 0.05

NUM_CLASSES = 5
WINDOW_SIZE = 168
NUM_FEATURES = 4

RANDOM_STATE = 42

# Class weights - exactly matching your FedAvg implementation
CLASS_WEIGHTS = {
    0: 3.0,
    1: 2.0,
    2: 1.0,
    3: 1.0,
    4: 1.0
}

# ============================================================
# 2. Seed
# ============================================================

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

print("=" * 70)
print("FedAvg + Pruning")
print("=" * 70)

# ============================================================
# 3. Loading Preprocessed Data
# ============================================================

print("\nStep 1: Loading Preprocessed Data")

X_nodes = []
y_nodes = []

for i in range(NUM_NODES):

    X_path = os.path.join(
        BASE_PATH,
        f"X_node_{i}_5class_timeseries168-1.npy"
    )

    y_path = os.path.join(
        BASE_PATH,
        f"y_node_{i}_5class_timeseries168-1.npy"
    )

    X_n = np.load(X_path)
    y_n = np.load(y_path)

    # Important validation
    if len(X_n) != len(y_n):
        raise ValueError(
            f"Mismatch between X and y in Node {i+1}: "
            f"X={len(X_n)}, y={len(y_n)}"
        )

    X_nodes.append(X_n.astype(np.float32))
    y_nodes.append(y_n.astype(np.int8))

    print(
        f"Node {i+1}: "
        f"X={X_n.shape}, "
        f"y={y_n.shape}, "
        f"distribution={np.bincount(y_n)}"
    )

# ============================================================
# 4. Loading the Central Test Set
# ============================================================

X_test = np.load(
    os.path.join(
        BASE_PATH,
        "X_test_5class_timeseries168-1.npy"
    )
).astype(np.float32)

y_test = np.load(
    os.path.join(
        BASE_PATH,
        "y_test_5class_timeseries168-1.npy"
    )
).astype(np.int8)

if len(X_test) != len(y_test):
    raise ValueError(
        f"Mismatch between X_test and y_test: "
        f"X_test={len(X_test)}, y_test={len(y_test)}"
    )

print("\nCentral Test:")
print(f"X_test = {X_test.shape}")
print(f"y_test = {y_test.shape}")
print(f"distribution = {np.bincount(y_test)}")

# ============================================================
# 5. Building the LSTM Model
#    Exactly the same architecture as the FedAvg model
# ============================================================

def create_model():

    model = tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=(WINDOW_SIZE, NUM_FEATURES)
        ),

        tf.keras.layers.LSTM(
            64,
            return_sequences=False
        ),

        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(
            16,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l1(0.01)
        ),

        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(
            NUM_CLASSES,
            activation='softmax'
        )
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ============================================================
# 6. Complete Evaluation Function
# ============================================================

def evaluate_model(model, X, y, title="Model"):

    print("\n" + "=" * 70)
    print(f"Evaluation: {title}")
    print("=" * 70)

    # Loss and Accuracy
    loss, accuracy = model.evaluate(
        X,
        y,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    # Prediction
    y_prob = model.predict(
        X,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    y_pred = np.argmax(y_prob, axis=1)

    # F1
    macro_f1 = f1_score(
        y,
        y_pred,
        average='macro',
        zero_division=0
    )

    weighted_f1 = f1_score(
        y,
        y_pred,
        average='weighted',
        zero_division=0
    )

    # Custom weighted F1
    class_f1 = f1_score(
        y,
        y_pred,
        average=None,
        labels=[0, 1, 2, 3, 4],
        zero_division=0
    )

    custom_weights = np.array([
        3.0, 2.0, 1.0, 1.0, 1.0
    ])

    custom_weighted_f1 = np.average(
        class_f1,
        weights=custom_weights
    )

    # RMSE / MAE on class labels from 0 to 4
    rmse = np.sqrt(
        mean_squared_error(y, y_pred)
    )

    mae = mean_absolute_error(
        y,
        y_pred
    )

    # --------------------------------------------------------
    # Main Results
    # --------------------------------------------------------

    print(f"\nLoss           : {loss:.4f}")
    print(f"Accuracy       : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Macro F1       : {macro_f1:.4f}")
    print(f"Weighted F1    : {weighted_f1:.4f}")
    print(f"Custom F1      : {custom_weighted_f1:.4f}")
    print(f"RMSE           : {rmse:.4f}")
    print(f"MAE            : {mae:.4f}")

    # --------------------------------------------------------
    # Classification Report
    # --------------------------------------------------------

    print("\nClassification Report:")
    print(
        classification_report(
            y,
            y_pred,
            labels=[0, 1, 2, 3, 4],
            target_names=[
                "Class 0",
                "Class 1",
                "Class 2",
                "Class 3",
                "Class 4"
            ],
            digits=4,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            y,
            y_pred,
            labels=[0, 1, 2, 3, 4]
        )
    )

    return {
        "loss": loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "custom_weighted_f1": custom_weighted_f1,
        "rmse": rmse,
        "mae": mae,
        "class_f1": class_f1
    }


# ============================================================
# 7. Starting FedAvg
# ============================================================

print("\n" + "=" * 70)
print("Starting One-Round FedAvg")
print("=" * 70)

# Initial Global Model
global_model = create_model()

global_weights = global_model.get_weights()

# To store the history
fedavg_history = []

# ============================================================
# 8. FedAvg - One Round Only
# ============================================================

for round_num in range(ROUNDS):

    print("\n" + "-" * 70)
    print(f"FedAvg Round {round_num + 1}/{ROUNDS}")
    print("-" * 70)

    local_weights = []
    client_sizes = []

    # --------------------------------------------------------
    # Training Local Models
    # --------------------------------------------------------

    for node_id in range(NUM_NODES):

        print(f"\nTraining Node {node_id + 1}/{NUM_NODES}")

        X_n = X_nodes[node_id]
        y_n = y_nodes[node_id]

        # The same exact split used in your FedAvg implementation
        X_train, X_local_test, y_train, y_local_test = train_test_split(
            X_n,
            y_n,
            test_size=0.2,
            random_state=42,
            stratify=y_n
        )

        print(f"Train samples : {len(X_train)}")
        print(f"Local test    : {len(X_local_test)}")

        # Local Model
        local_model = create_model()

        # Initialize with Global weights
        local_model.set_weights(global_weights)

        # Training
        history = local_model.fit(
            X_train,
            y_train,
            epochs=LOCAL_EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=CLASS_WEIGHTS,
            shuffle=True,
            verbose=1
        )

        # Save Local weights
        local_weights.append(
            local_model.get_weights()
        )

        # The same client size used in your FedAvg implementation
        client_sizes.append(
            len(y_train)
        )

        # Freeing memory
        del (
            local_model,
            X_train,
            X_local_test,
            y_train,
            y_local_test,
            X_n,
            y_n
        )

        gc.collect()

        tf.keras.backend.clear_session()

    # ========================================================
    # 9. FedAvg Weighted Aggregation
    # ========================================================

    print("\nAggregating Local Models using Weighted FedAvg...")

    total_samples = np.sum(client_sizes)

    print(f"Client sizes: {client_sizes}")
    print(f"Total train samples: {total_samples}")

    # Creating new Global weights
    new_global_weights = []

    for layer_weights in zip(*local_weights):

        weighted_sum = np.zeros_like(
            layer_weights[0],
            dtype=np.float64
        )

        for client_weight, client_size in zip(
            layer_weights,
            client_sizes
        ):

            weighted_sum += (
                client_weight.astype(np.float64)
                * (client_size / total_samples)
            )

        new_global_weights.append(
            weighted_sum.astype(np.float32)
        )

    # Updating the Global model
    global_weights = new_global_weights

    global_model.set_weights(global_weights)

    # ========================================================
    # 10. Evaluating Global FedAvg Before Pruning
    # ========================================================

    round_results = evaluate_model(
        global_model,
        X_test,
        y_test,
        title=f"FedAvg Round {round_num + 1} - Before Pruning"
    )

    fedavg_history.append(round_results)

# ============================================================
# 11. Counting Parameters Before Pruning
# ============================================================

print("\n" + "=" * 70)
print("Model Parameters Before Pruning")
print("=" * 70)

total_params = 0
nonzero_before = 0

for w in global_model.get_weights():

    total_params += w.size
    nonzero_before += np.count_nonzero(w)

print(f"Total parameters       : {total_params:,}")
print(f"Non-zero parameters    : {nonzero_before:,}")

# ============================================================
# 12. Pruning Function
# ============================================================

def prune_model_weights(model, threshold=0.05):

    print("\n" + "=" * 70)
    print(f"Applying Magnitude Pruning | Threshold = {threshold}")
    print("=" * 70)

    weights = model.get_weights()

    total_params = 0
    nonzero_before = 0
    nonzero_after = 0

    pruned_weights = []

    for w in weights:

        total_params += w.size

        # Number of non-zero weights before pruning
        nonzero_before += np.count_nonzero(w)

        # Copying weights
        w_pruned = w.copy()

        # Pruning mask
        mask = np.abs(w_pruned) < threshold

        # Setting small weights to zero
        w_pruned[mask] = 0.0

        # Number of non-zero weights after pruning
        nonzero_after += np.count_nonzero(w_pruned)

        pruned_weights.append(w_pruned)

    # Applying pruned weights
    model.set_weights(pruned_weights)

    # Calculations
    removed_weights = (
        nonzero_before - nonzero_after
    )

    reduction_percent = (
        removed_weights / nonzero_before
    ) * 100.0

    zero_after = (
        total_params - nonzero_after
    )

    sparsity_percent = (
        zero_after / total_params
    ) * 100.0

    return {
        "total_params": total_params,
        "nonzero_before": nonzero_before,
        "nonzero_after": nonzero_after,
        "removed_weights": removed_weights,
        "reduction_percent": reduction_percent,
        "zero_after": zero_after,
        "sparsity_percent": sparsity_percent
    }


# ============================================================
# 13. Applying Pruning
# ============================================================

pruning_results = prune_model_weights(
    global_model,
    threshold=PRUNING_THRESHOLD
)

# ============================================================
# 14. Displaying Pruning Results
# ============================================================

print("\n" + "=" * 70)
print("Pruning Results")
print("=" * 70)

print(
    f"Total parameters       : "
    f"{pruning_results['total_params']:,}"
)

print(
    f"Non-zero before        : "
    f"{pruning_results['nonzero_before']:,}"
)

print(
    f"Non-zero after         : "
    f"{pruning_results['nonzero_after']:,}"
)

print(
    f"Removed weights        : "
    f"{pruning_results['removed_weights']:,}"
)

print(
    f"Active-weight reduction: "
    f"{pruning_results['reduction_percent']:.2f}%"
)

print(
    f"Zero weights after     : "
    f"{pruning_results['zero_after']:,}"
)

print(
    f"Sparsity               : "
    f"{pruning_results['sparsity_percent']:.2f}%"
)

# ============================================================
# 15. Evaluating the Model After Pruning
# ============================================================

pruned_results = evaluate_model(
    global_model,
    X_test,
    y_test,
    title="FedAvg + Pruning - Final"
)

# ============================================================
# 16. Comparison Before and After Pruning
# ============================================================

before_results = fedavg_history[-1]

print("\n" + "=" * 70)
print("FedAvg Before vs FedAvg + Pruning")
print("=" * 70)

print(
    f"{'Metric':<25}"
    f"{'Before Pruning':>20}"
    f"{'After Pruning':>20}"
)

print("-" * 70)

print(
    f"{'Accuracy':<25}"
    f"{before_results['accuracy']:>20.4f}"
    f"{pruned_results['accuracy']:>20.4f}"
)

print(
    f"{'Macro F1':<25}"
    f"{before_results['macro_f1']:>20.4f}"
    f"{pruned_results['macro_f1']:>20.4f}"
)

print(
    f"{'Weighted F1':<25}"
    f"{before_results['weighted_f1']:>20.4f}"
    f"{pruned_results['weighted_f1']:>20.4f}"
)

print(
    f"{'Custom Weighted F1':<25}"
    f"{before_results['custom_weighted_f1']:>20.4f}"
    f"{pruned_results['custom_weighted_f1']:>20.4f}"
)

print(
    f"{'RMSE':<25}"
    f"{before_results['rmse']:>20.4f}"
    f"{pruned_results['rmse']:>20.4f}"
)

print(
    f"{'MAE':<25}"
    f"{before_results['mae']:>20.4f}"
    f"{pruned_results['mae']:>20.4f}"
)

# ============================================================
# 17. Changes in Accuracy and F1
# ============================================================

accuracy_change = (
    pruned_results["accuracy"]
    - before_results["accuracy"]
)

macro_f1_change = (
    pruned_results["macro_f1"]
    - before_results["macro_f1"]
)

weighted_f1_change = (
    pruned_results["weighted_f1"]
    - before_results["weighted_f1"]
)

print("\nChanges after pruning:")

print(
    f"Accuracy change    : "
    f"{accuracy_change:+.4f} "
    f"({accuracy_change*100:+.2f} percentage points)"
)

print(
    f"Macro F1 change    : "
    f"{macro_f1_change:+.4f}"
)

print(
    f"Weighted F1 change : "
    f"{weighted_f1_change:+.4f}"
)

# ============================================================
# 18. Saving the Final Model
# ============================================================

MODEL_PATH = os.path.join(
    BASE_PATH,
    "FedAvg_Pruned_LSTM_5class_timeseries168_1round.keras"
)

global_model.save(MODEL_PATH)

print("\n" + "=" * 70)
print("Final model saved:")
print(MODEL_PATH)
print("=" * 70)

# ============================================================
# 19. Saving Results to a File
# ============================================================

RESULTS_PATH = os.path.join(
    BASE_PATH,
    "FedAvg_Pruning_results_1round.txt"
)

with open(RESULTS_PATH, "w", encoding="utf-8") as f:

    f.write("FedAvg + Pruning Results\n")
    f.write("=" * 60 + "\n\n")

    f.write("Configuration\n")
    f.write("-" * 60 + "\n")
    f.write(f"Nodes: {NUM_NODES}\n")
    f.write(f"Rounds: {ROUNDS}\n")
    f.write(f"Local Epochs: {LOCAL_EPOCHS}\n")
    f.write(f"Batch Size: {BATCH_SIZE}\n")
    f.write(f"Learning Rate: {LEARNING_RATE}\n")
    f.write(f"Pruning Threshold: {PRUNING_THRESHOLD}\n\n")

    f.write("Before Pruning\n")
    f.write("-" * 60 + "\n")
    f.write(
        f"Accuracy: "
        f"{before_results['accuracy']:.4f}\n"
    )
    f.write(
        f"Macro F1: "
        f"{before_results['macro_f1']:.4f}\n"
    )
    f.write(
        f"Weighted F1: "
        f"{before_results['weighted_f1']:.4f}\n"
    )
    f.write(
        f"Custom Weighted F1: "
        f"{before_results['custom_weighted_f1']:.4f}\n"
    )
    f.write(
        f"RMSE: "
        f"{before_results['rmse']:.4f}\n"
    )
    f.write(
        f"MAE: "
        f"{before_results['mae']:.4f}\n"
    )

    f.write("\nPruning\n")
    f.write("-" * 60 + "\n")

    f.write(
        f"Total Parameters: "
        f"{pruning_results['total_params']}\n"
    )

    f.write(
        f"Non-zero Before: "
        f"{pruning_results['nonzero_before']}\n"
    )

    f.write(
        f"Non-zero After: "
        f"{pruning_results['nonzero_after']}\n"
    )

    f.write(
        f"Removed Weights: "
        f"{pruning_results['removed_weights']}\n"
    )

    f.write(
        f"Active-weight Reduction: "
        f"{pruning_results['reduction_percent']:.2f}%\n"
    )

    f.write(
        f"Sparsity: "
        f"{pruning_results['sparsity_percent']:.2f}%\n"
    )

    f.write("\nAfter Pruning\n")
    f.write("-" * 60 + "\n")

    f.write(
        f"Accuracy: "
        f"{pruned_results['accuracy']:.4f}\n"
    )

    f.write(
        f"Macro F1: "
        f"{pruned_results['macro_f1']:.4f}\n"
    )

    f.write(
        f"Weighted F1: "
        f"{pruned_results['weighted_f1']:.4f}\n"
    )

    f.write(
        f"Custom Weighted F1: "
        f"{pruned_results['custom_weighted_f1']:.4f}\n"
    )

    f.write(
        f"RMSE: "
        f"{pruned_results['rmse']:.4f}\n"
    )

    f.write(
        f"MAE: "
        f"{pruned_results['mae']:.4f}\n"
    )

print(f"\nResults saved to the following file:")
print(RESULTS_PATH)

# ============================================================
# 20. End
# ============================================================

print("\n" + "=" * 70)
print("FedAvg + Pruning completed successfully.")
print("=" * 70)
