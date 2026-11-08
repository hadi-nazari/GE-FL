# ============================================================
# CENTRALIZED LSTM + GA + MAGNITUDE-BASED PRUNING
# 5-class PM2.5 | 168-hour input | 24-hour forecasting horizon
# ============================================================

import os
import random
import numpy as np
import tensorflow as tf

from deap import base, creator, tools
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
# 1. Reproducibility and paths
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_PATH = "/content/drive/My Drive/beijing_data/"

X_TRAIN_PATH = os.path.join(
    BASE_PATH, "X_centralized_5class_timeseries168-1.npy"
)
Y_TRAIN_PATH = os.path.join(
    BASE_PATH, "y_centralized_5class_timeseries168-1.npy"
)
X_TEST_PATH = os.path.join(
    BASE_PATH, "X_test_centralized_5class_timeseries168-1.npy"
)
Y_TEST_PATH = os.path.join(
    BASE_PATH, "y_test_centralized_5class_timeseries168-1.npy"
)

MODEL_BASE_PATH = os.path.join(
    BASE_PATH, "Centralized_LSTM_5class_168_24.keras"
)
MODEL_GA_PATH = os.path.join(
    BASE_PATH, "Centralized_LSTM_GA_Pruned_5class_168_24.keras"
)
WEIGHTS_GA_PATH = os.path.join(
    BASE_PATH, "Centralized_LSTM_GA_Pruned_5class_168_24.weights.h5"
)
GA_HISTORY_PATH = os.path.join(
    BASE_PATH, "Centralized_GA_accuracies_168.npy"
)

# ============================================================
# 2. Experiment configuration
# ============================================================

EPOCHS = 10
BATCH_SIZE = 128

POP_SIZE = 20
GENERATIONS = 25
COMPRESSION_THRESHOLD = 0.05

CROSSOVER_PROB = 0.8
MUTATION_PROB = 0.5
MUTATION_SIGMA = 0.1
GENE_MUTATION_PROB = 0.01
TOURNAMENT_SIZE = 3

CLASS_WEIGHTS = {
    0: 3.0,
    1: 2.0,
    2: 1.0,
    3: 1.0,
    4: 1.0
}
CUSTOM_F1_WEIGHTS = np.array([3, 2, 1, 1, 1], dtype=np.float64)
LABELS = np.arange(5)

# ============================================================
# 3. Load centralized data
# ============================================================

X = np.load(X_TRAIN_PATH).astype(np.float32)
y = np.load(Y_TRAIN_PATH).astype(np.int32)

X_test = np.load(X_TEST_PATH).astype(np.float32)
y_test = np.load(Y_TEST_PATH).astype(np.int32)

assert len(X) == len(y), "Training X/y lengths do not match."
assert len(X_test) == len(y_test), "Test X/y lengths do not match."
assert X.ndim == 3 and X.shape[1:] == (168, 4), X.shape
assert X_test.ndim == 3 and X_test.shape[1:] == (168, 4), X_test.shape

print("Centralized training:", X.shape, y.shape)
print("Centralized test    :", X_test.shape, y_test.shape)

# Same train/validation split as the centralized baseline.
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)

print("Train      :", X_train.shape)
print("Validation :", X_val.shape)

# ============================================================
# 4. Identical LSTM architecture
# ============================================================

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(168, 4)),
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(
            16,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l1(0.01)
        ),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(5, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ============================================================
# 5. Train centralized baseline
# ============================================================

baseline_model = create_model()
baseline_model.summary()

baseline_model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=CLASS_WEIGHTS,
    shuffle=True,
    verbose=1
)

# Save the centralized model before GA.
baseline_model.save(MODEL_BASE_PATH)

# Preserve the trained weights as the GA initialization.
initial_weights = [
    np.array(w, copy=True) for w in baseline_model.get_weights()
]

# ============================================================
# 6. Weight-vector utilities
# ============================================================

def flatten_weights(weights):
    return np.concatenate(
        [w.reshape(-1) for w in weights]
    ).astype(np.float32)

def restore_weights(vector, template_weights):
    restored = []
    offset = 0

    for weight in template_weights:
        size = weight.size
        restored.append(
            np.asarray(
                vector[offset:offset + size],
                dtype=weight.dtype
            ).reshape(weight.shape)
        )
        offset += size

    if offset != len(vector):
        raise ValueError(
            f"Chromosome length mismatch: {len(vector)} vs {offset}"
        )

    return restored

initial_vector = flatten_weights(initial_weights)
TOTAL_PARAMS = len(initial_vector)

print("\nTotal architecture parameters:", TOTAL_PARAMS)

# Separate model for candidate evaluation.
ga_model = create_model()

# ============================================================
# 7. Fitness evaluation
# Uses X_test and y_test, as requested.
# ============================================================

def evaluate_individual(individual):
    candidate = np.asarray(individual, dtype=np.float32).copy()

    # Magnitude-based unstructured pruning
    candidate[np.abs(candidate) < COMPRESSION_THRESHOLD] = 0.0

    candidate_weights = restore_weights(
        candidate, initial_weights
    )
    ga_model.set_weights(candidate_weights)

    probabilities = ga_model.predict(
        X_test, batch_size=512, verbose=0
    )
    predictions = np.argmax(probabilities, axis=1)

    accuracy = accuracy_score(y_test, predictions)

    per_class_f1 = f1_score(
        y_test,
        predictions,
        labels=LABELS,
        average=None,
        zero_division=0
    )

    custom_weighted_f1 = np.average(
        per_class_f1,
        weights=CUSTOM_F1_WEIGHTS
    )

    nonzero_fraction = (
        np.count_nonzero(candidate) / len(candidate)
    )

    fitness = (
        0.5 * accuracy
        + 0.4 * custom_weighted_f1
        - 0.1 * nonzero_fraction
    )

    return (float(fitness),)

# ============================================================
# 8. DEAP setup
# ============================================================

# Unique names prevent errors if this cell is rerun.
if not hasattr(creator, "CentralizedGAFitnessMax"):
    creator.create(
        "CentralizedGAFitnessMax",
        base.Fitness,
        weights=(1.0,)
    )

if not hasattr(creator, "CentralizedGAIndividual"):
    creator.create(
        "CentralizedGAIndividual",
        list,
        fitness=creator.CentralizedGAFitnessMax
    )

toolbox = base.Toolbox()

def initialize_individual():
    # Initialize around trained centralized weights.
    perturbed = initial_vector + np.random.normal(
        loc=0.0,
        scale=0.01,
        size=initial_vector.shape
    ).astype(np.float32)

    return creator.CentralizedGAIndividual(
        perturbed.tolist()
    )

toolbox.register("individual", initialize_individual)
toolbox.register(
    "population", tools.initRepeat, list, toolbox.individual
)
toolbox.register("evaluate", evaluate_individual)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register(
    "mutate",
    tools.mutGaussian,
    mu=0.0,
    sigma=MUTATION_SIGMA,
    indpb=GENE_MUTATION_PROB
)
toolbox.register(
    "select",
    tools.selTournament,
    tournsize=TOURNAMENT_SIZE
)

# ============================================================
# 9. Initialize population and run GA
# ============================================================

population = toolbox.population(n=POP_SIZE)

# Keep the original centralized solution in the initial population.
population[0] = creator.CentralizedGAIndividual(
    initial_vector.tolist()
)

for individual in population:
    individual.fitness.values = toolbox.evaluate(individual)

hall_of_fame = tools.HallOfFame(1)
hall_of_fame.update(population)

accuracies_per_gen = []
fitness_per_gen = []
nonzero_per_gen = []
sparsity_per_gen = []

for generation in range(GENERATIONS):

    offspring = list(map(
        toolbox.clone,
        toolbox.select(population, len(population))
    ))

    # One-point crossover
    for child1, child2 in zip(
        offspring[::2], offspring[1::2]
    ):
        if random.random() < CROSSOVER_PROB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    # Gaussian mutation
    for mutant in offspring:
        if random.random() < MUTATION_PROB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Evaluate modified individuals
    invalid_individuals = [
        ind for ind in offspring if not ind.fitness.valid
    ]

    for individual in invalid_individuals:
        individual.fitness.values = toolbox.evaluate(individual)

    population = offspring
    hall_of_fame.update(population)

    # Evaluate best individual of this generation.
    best_vector = np.asarray(
        hall_of_fame[0], dtype=np.float32
    ).copy()
    best_vector[
        np.abs(best_vector) < COMPRESSION_THRESHOLD
    ] = 0.0

    ga_model.set_weights(
        restore_weights(best_vector, initial_weights)
    )

    best_probabilities = ga_model.predict(
        X_test, batch_size=512, verbose=0
    )
    best_predictions = np.argmax(
        best_probabilities, axis=1
    )

    generation_accuracy = accuracy_score(
        y_test, best_predictions
    )
    generation_fitness = hall_of_fame[0].fitness.values[0]

    nonzero = int(np.count_nonzero(best_vector))
    sparsity = 1.0 - nonzero / TOTAL_PARAMS

    accuracies_per_gen.append(generation_accuracy)
    fitness_per_gen.append(generation_fitness)
    nonzero_per_gen.append(nonzero)
    sparsity_per_gen.append(sparsity)

    print(
        f"Generation {generation + 1:02d}/{GENERATIONS} | "
        f"Test Accuracy={generation_accuracy:.4f} | "
        f"Fitness={generation_fitness:.4f} | "
        f"Non-zero={nonzero} | "
        f"Sparsity={sparsity:.2%}"
    )

# ============================================================
# 10. Construct final GA-pruned model
# ============================================================

best_vector = np.asarray(
    hall_of_fame[0], dtype=np.float32
).copy()

best_vector[
    np.abs(best_vector) < COMPRESSION_THRESHOLD
] = 0.0

final_model = create_model()
final_model.set_weights(
    restore_weights(best_vector, initial_weights)
)

# ============================================================
# 11. Final evaluation
# ============================================================

test_loss, _ = final_model.evaluate(
    X_test, y_test, batch_size=512, verbose=0
)

y_prob = final_model.predict(
    X_test, batch_size=512, verbose=0
)
y_pred = np.argmax(y_prob, axis=1)

accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(
    y_test, y_pred, average="macro", zero_division=0
)
weighted_f1 = f1_score(
    y_test, y_pred, average="weighted", zero_division=0
)

per_class_f1 = f1_score(
    y_test,
    y_pred,
    labels=LABELS,
    average=None,
    zero_division=0
)

custom_f1 = np.average(
    per_class_f1,
    weights=CUSTOM_F1_WEIGHTS
)

# RMSE/MAE here are for class labels (0-4), not concentration units.
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

nonzero_after = int(np.count_nonzero(best_vector))
removed_params = TOTAL_PARAMS - nonzero_after
sparsity = removed_params / TOTAL_PARAMS

print("\n" + "=" * 65)
print("CENTRALIZED LSTM + GA + MAGNITUDE PRUNING")
print("=" * 65)
print(f"Test Loss             : {test_loss:.4f}")
print(f"Accuracy              : {accuracy:.4f}")
print(f"Macro F1              : {macro_f1:.4f}")
print(f"Weighted F1           : {weighted_f1:.4f}")
print(f"Custom Weighted F1    : {custom_f1:.4f}")
print(f"Class-label RMSE      : {rmse:.4f}")
print(f"Class-label MAE       : {mae:.4f}")
print(f"Total Parameters      : {TOTAL_PARAMS}")
print(f"Non-zero Parameters   : {nonzero_after}")
print(f"Removed Parameters    : {removed_params}")
print(f"Sparsity              : {sparsity:.2%}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=LABELS,
    digits=4,
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=LABELS))

# ============================================================
# 12. Save outputs
# ============================================================

final_model.save(MODEL_GA_PATH)
final_model.save_weights(WEIGHTS_GA_PATH)

np.save(
    GA_HISTORY_PATH,
    np.asarray(accuracies_per_gen, dtype=np.float32)
)

np.save(
    os.path.join(BASE_PATH, "Centralized_GA_fitness_168.npy"),
    np.asarray(fitness_per_gen, dtype=np.float32)
)

print("\nSaved model:", MODEL_GA_PATH)
print("Saved weights:", WEIGHTS_GA_PATH)
print("Saved generation accuracies:", GA_HISTORY_PATH)
