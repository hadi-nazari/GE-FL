
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import ADASYN
import os
import gc

BASE_PATH = "/content/drive/My Drive/beijing_data/"

def create_time_series_data(data, window_size=168, horizon=24):
    X, y = [], []
    features = ['PM2.5', 'CO', 'TEMP', 'DEWP']
    data_array = data[features].values.astype(np.float32)  
    for i in range(len(data) - window_size - horizon + 1):
        X.append(data_array[i:i+window_size])
        y.append(pm25_to_class(data['PM2.5'].iloc[i + window_size + horizon - 1]))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int8)

def pm25_to_class(pm25):
    if pm25 <= 12:
        return 0
    elif pm25 <= 35.4:
        return 1
    elif pm25 <= 55.4:
        return 2
    elif pm25 <= 150.4:
        return 3
    else:
        return 4

def load_and_prepare_data():
    print("\nStep 1: Dataset Loading and Preprocessing (Memory-Efficient Version + ADASYN)")
    global NUM_NODES
    NUM_NODES = 5

    try:
        data = pd.read_csv(os.path.join(BASE_PATH, "merged_beijing_data.csv"))
    except FileNotFoundError:
        raise FileNotFoundError(f"File merged_beijing_data.csv not found in {BASE_PATH}.")

    print(f"Number of raw rows: {len(data)}")

    data['datetime'] = pd.to_datetime(data[['year', 'month', 'day', 'hour']])
    data = data.sort_values('datetime').drop(columns=['datetime'])

    # 
    data = data[data['PM2.5'].between(2, 500)]
    data = data[data['CO'].between(100, 5000)]
    print(f"Number of rows after removing outliers: {len(data)}")

    features = ['PM2.5', 'CO', 'TEMP', 'DEWP']
    data[features] = data[features].fillna(data[features].mean())

    X, y = create_time_series_data(data)
    print(f"Time-series data shape: X={X.shape}, y={y.shape}")
    print(f"Class distribution before duplicate removal: {np.bincount(y)}")

    X_flat = X.reshape(X.shape[0], -1)
    Xy = np.hstack((X_flat, y.reshape(-1, 1)))
    _, indices = np.unique(Xy, axis=0, return_index=True)
    indices = np.sort(indices)
    X = X[indices]
    y = y[indices]
    del X_flat, Xy
    gc.collect()

    print(f"Number of unique samples: {len(X)}")
    print(f"Class distribution after duplicate removal: {np.bincount(y)}")

    # ====================== Chronological Split ======================
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    del X, y
    gc.collect()

    print(f"\nChronological split completed:")
    print(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples")
    print(f"Class distribution in y_train: {np.bincount(y_train)}")
    print(f"Class distribution in y_test: {np.bincount(y_test)}")

    scaler = StandardScaler()
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat  = X_test.reshape(X_test.shape[0], -1)

    X_train_scaled = scaler.fit_transform(X_train_flat).astype(np.float32).reshape(X_train.shape)
    X_test_scaled  = scaler.transform(X_test_flat).astype(np.float32).reshape(X_test.shape)

    del X_train, X_train_flat, X_test_flat
    gc.collect()

    X_nodes = np.array_split(X_train_scaled, NUM_NODES)
    y_nodes = np.array_split(y_train, NUM_NODES)
    del X_train_scaled, y_train
    gc.collect()

    X_nodes_balanced, y_nodes_balanced = [], []

    adasyn = ADASYN(
        random_state=42,
        sampling_strategy='minority',   #
        n_neighbors=3
    )

    for i, (X_n, y_n) in enumerate(zip(X_nodes, y_nodes)):
        print(f"\nNode {i+1} before balancing: {len(X_n)} samples, distribution: {np.bincount(y_n)}")

        if len(np.unique(y_n)) > 1 and len(X_n) >= 5:
            try:
                X_n_flat = X_n.reshape(X_n.shape[0], -1)
                X_n_bal, y_n_bal = adasyn.fit_resample(X_n_flat, y_n)
                X_n_bal = X_n_bal.astype(np.float32).reshape(-1, X_n.shape[1], X_n.shape[2])
                y_n_bal = y_n_bal.astype(np.int8)

                X_nodes_balanced.append(X_n_bal)
                y_nodes_balanced.append(y_n_bal)
                print(f"Node {i+1} after balancing: {len(X_n_bal)} samples, distribution: {np.bincount(y_n_bal)}")

                del X_n_flat, X_n_bal, y_n_bal
            except Exception as e:
                print(f"Error in Balance{i+1}: {e}")
                X_nodes_balanced.append(X_n)
                y_nodes_balanced.append(y_n)
        else:
            X_nodes_balanced.append(X_n)
            y_nodes_balanced.append(y_n)

        del X_n, y_n
        gc.collect()

    np.save(os.path.join(BASE_PATH, 'X_test_5class_timeseries168-1.npy'), X_test_scaled)
    np.save(os.path.join(BASE_PATH, 'y_test_5class_timeseries168-1.npy'), y_test)

    for i, (X_n, y_n) in enumerate(zip(X_nodes_balanced, y_nodes_balanced)):
        np.save(os.path.join(BASE_PATH, f'X_node_{i}_5class_timeseries168-1.npy'), X_n)
        np.save(os.path.join(BASE_PATH, f'y_node_{i}_5class_timeseries168-1.npy'), y_n)

    print("\nData successfully saved.")
    return list(X_nodes_balanced), list(y_nodes_balanced), X_test_scaled, y_test, scaler


if __name__ == "__main__":
    np.random.seed(42)
    X_nodes, y_nodes, X_test, y_test, scaler = load_and_prepare_data()
