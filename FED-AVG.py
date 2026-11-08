
import os
import gc
import random
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# Configuration
# ============================================================

BASE_PATH = "/content/drive/My Drive/beijing_data"

NUM_NODES = 5
ROUNDS = 1
LOCAL_EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.001

RANDOM_STATE = 42

# همان class weights که در Local Training استفاده کردی
CLASS_WEIGHTS = {
    0: 3.0,
    1: 2.0,
    2: 1.0,
    3: 1.0,
    4: 1.0
}


# ============================================================
# Reproducibility
# ============================================================

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ============================================================
# Create Model
# ============================================================

def create_local_model():

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(168, 4)),

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
            5,
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
# Load Data
# ============================================================

def load_data():

    print("\nLoading federated datasets...")

    X_nodes = []
    y_nodes = []

    for i in range(NUM_NODES):

        x_path = os.path.join(
            BASE_PATH,
            f'X_node_{i}_5class_timeseries168-1.npy'
        )

        y_path = os.path.join(
            BASE_PATH,
            f'y_node_{i}_5class_timeseries168-1.npy'
        )

        X = np.load(x_path)
        y = np.load(y_path)

        X_nodes.append(X)
        y_nodes.append(y)

        print(
            f"Node {i+1}: "
            f"X={X.shape}, "
            f"y={y.shape}, "
            f"class distribution={np.bincount(y)}"
        )

    X_test = np.load(
        os.path.join(
            BASE_PATH,
            'X_test_5class_timeseries168-1.npy'
        )
    )

    y_test = np.load(
        os.path.join(
            BASE_PATH,
            'y_test_5class_timeseries168-1.npy'
        )
    )

    print(
        f"\nCentral Test: "
        f"X={X_test.shape}, "
        f"y={y_test.shape}"
    )

    return X_nodes, y_nodes, X_test, y_test


# ============================================================
# Local Training
# ============================================================

def train_local_model(global_weights, X, y):

    model = create_local_model()

    # دریافت وزن‌های Global
    model.set_weights(global_weights)

    # --------------------------------------------------------
    # Local training
    # --------------------------------------------------------

    model.fit(
        X,
        y,
        epochs=LOCAL_EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        class_weight=CLASS_WEIGHTS,
        shuffle=True
    )

    local_weights = model.get_weights()

    del model
    gc.collect()

    return local_weights


# ============================================================
# FedAvg
# ============================================================

def fedavg(local_weights_list, client_sizes):

    total_samples = np.sum(client_sizes)

    global_weights = []

    for layer_weights in zip(*local_weights_list):

        weighted_sum = np.zeros_like(
            layer_weights[0],
            dtype=np.float64
        )

        for weights, n_samples in zip(
            layer_weights,
            client_sizes
        ):

            weighted_sum += (
                weights.astype(np.float64)
                * (n_samples / total_samples)
            )

        global_layer = weighted_sum.astype(
            layer_weights[0].dtype
        )

        global_weights.append(global_layer)

    return global_weights


# ============================================================
# Evaluate Global Model
# ============================================================

def evaluate_global_model(
    global_model,
    X_test,
    y_test,
    round_number
):

    loss, accuracy = global_model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    y_pred_proba = global_model.predict(
        X_test,
        verbose=0
    )

    y_pred = np.argmax(
        y_pred_proba,
        axis=1
    )

    # --------------------------------------------------------
    # F1 scores
    # --------------------------------------------------------

    f1_per_class = f1_score(
        y_test,
        y_pred,
        average=None,
        labels=[0, 1, 2, 3, 4],
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average='macro',
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average='weighted',
        zero_division=0
    )

    # همان weighted F1 سفارشی تو
    custom_weighted_f1 = (
        3.0 * f1_per_class[0] +
        2.0 * f1_per_class[1] +
        1.0 * f1_per_class[2] +
        1.0 * f1_per_class[3] +
        1.0 * f1_per_class[4]
    ) / 8.0

    print(
        f"\nRound {round_number}"
    )

    print(
        f"Global Loss      : {loss:.4f}"
    )

    print(
        f"Global Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Macro F1         : {macro_f1:.4f}"
    )

    print(
        f"Weighted F1      : {weighted_f1:.4f}"
    )

    print(
        f"Custom Weighted F1: "
        f"{custom_weighted_f1:.4f}"
    )

    print(
        "F1 per class:",
        np.round(f1_per_class, 4)
    )

    return {
        "round": round_number,
        "loss": loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "custom_weighted_f1": custom_weighted_f1
    }


# ============================================================
# Federated Training
# ============================================================

def run_fedavg():

    print("\n")
    print("=" * 70)
    print("FedAvg Federated Learning")
    print("=" * 70)

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    X_nodes, y_nodes, X_test, y_test = load_data()

    # --------------------------------------------------------
    # Client sizes
    # --------------------------------------------------------

    client_sizes = [
        len(y)
        for y in y_nodes
    ]

    print("\nClient sizes:")

    for i, size in enumerate(client_sizes):
        print(
            f"Node {i+1}: {size} samples"
        )

    print(
        f"Total samples: {sum(client_sizes)}"
    )

    # --------------------------------------------------------
    # Create initial global model
    # --------------------------------------------------------

    global_model = create_local_model()

    global_weights = global_model.get_weights()

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    history = []

    # --------------------------------------------------------
    # Federated Rounds
    # --------------------------------------------------------

    for round_number in range(1, ROUNDS + 1):

        print("\n")
        print("=" * 70)
        print(
            f"FEDERATED ROUND {round_number}/{ROUNDS}"
        )
        print("=" * 70)

        local_weights_list = []

        # ====================================================
        # Local Training
        # ====================================================

        for node_id in range(NUM_NODES):

            print(
                f"\nTraining Node {node_id + 1}..."
            )

            X_n = X_nodes[node_id]
            y_n = y_nodes[node_id]

            local_weights = train_local_model(
                global_weights,
                X_n,
                y_n
            )

            local_weights_list.append(
                local_weights
            )

            print(
                f"Node {node_id + 1} completed."
            )

        # ====================================================
        # FedAvg Aggregation
        # ====================================================

        print("\nPerforming FedAvg aggregation...")

        global_weights = fedavg(
            local_weights_list,
            client_sizes
        )

        # ----------------------------------------------------
        # Update Global Model
        # ----------------------------------------------------

        global_model.set_weights(
            global_weights
        )

        # ====================================================
        # Evaluate Global Model
        # ====================================================

        metrics = evaluate_global_model(
            global_model,
            X_test,
            y_test,
            round_number
        )

        history.append(metrics)

        # ----------------------------------------------------
        # Free memory
        # ----------------------------------------------------

        del local_weights_list
        gc.collect()

    # ========================================================
    # Final Evaluation
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FINAL GLOBAL MODEL")
    print("=" * 70)

    loss, accuracy = global_model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    y_pred_proba = global_model.predict(
        X_test,
        verbose=0
    )

    y_pred = np.argmax(
        y_pred_proba,
        axis=1
    )

    print(
        f"\nFinal Loss     : {loss:.4f}"
    )

    print(
        f"Final Accuracy : {accuracy:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1, 2, 3, 4],
            target_names=[
                "Class 0",
                "Class 1",
                "Class 2",
                "Class 3",
                "Class 4"
            ],
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1, 2, 3, 4]
        )
    )

    # ========================================================
    # Save Global Model
    # ========================================================

    global_model_path = os.path.join(
        BASE_PATH,
        "fedavg_global_model_5class_timeseries168.weights.h5"
    )

    global_model.save_weights(
        global_model_path
    )

    print(
        f"\nGlobal model saved to:\n"
        f"{global_model_path}"
    )

    # ========================================================
    # Save History
    # ========================================================

    history_df = pd.DataFrame(history)

    history_path = os.path.join(
        BASE_PATH,
        "fedavg_history_timeseries168.csv"
    )

    history_df.to_csv(
        history_path,
        index=False
    )

    print(
        f"FedAvg history saved to:\n"
        f"{history_path}"
    )

    # Accuracy history
    np.save(
        os.path.join(
            BASE_PATH,
            "fedavg_accuracy_per_round168.npy"
        ),
        history_df["accuracy"].values
    )

    # F1 history
    np.save(
        os.path.join(
            BASE_PATH,
            "fedavg_macro_f1_per_round168.npy"
        ),
        history_df["macro_f1"].values
    )

    return global_model, history_df


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    global_model, history = run_fedavg()

