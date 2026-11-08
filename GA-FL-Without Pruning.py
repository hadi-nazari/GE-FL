import os
import random
import numpy as np
import tensorflow as tf

from deap import base, creator, tools
from sklearn.metrics import accuracy_score, f1_score


# =========================================================
# 1. Configuration
# =========================================================

BASE_PATH = "/content/drive/My Drive/beijing_data/"

POP_SIZE = 20
GENERATIONS = 25

CROSSOVER_PROB = 0.8
MUTATION_APPLY_PROB = 0.5
MUTATION_INDPB = 0.01
MUTATION_SIGMA = 0.1

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# =========================================================
# 2. LSTM architecture (unchanged)
# =========================================================

def create_local_model():
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


# =========================================================
# 3. Flattening and restoring model weights
# =========================================================

def get_weights(model):
    return np.concatenate([
        w.flatten() for w in model.get_weights()
    ])


def set_weights(model, weights):
    weights = np.asarray(weights)
    new_weights = []
    idx = 0

    for w in model.get_weights():
        size = w.size
        new_weights.append(
            weights[idx:idx + size].reshape(w.shape)
        )
        idx += size

    if idx != len(weights):
        raise ValueError(
            f"Weight mismatch: expected {idx}, received {len(weights)}"
        )

    model.set_weights(new_weights)


# =========================================================
# 4. Initialize individuals from local models
# =========================================================

if not hasattr(creator, "FitnessMaxNoPruning"):
    creator.create(
        "FitnessMaxNoPruning",
        base.Fitness,
        weights=(1.0,)
    )

if not hasattr(creator, "IndividualNoPruning"):
    creator.create(
        "IndividualNoPruning",
        list,
        fitness=creator.FitnessMaxNoPruning
    )


def init_individual(local_models):
    selected_model = random.choice(local_models)
    weights = get_weights(selected_model)

    return creator.IndividualNoPruning(weights.tolist())


# =========================================================
# 5. Fitness evaluation WITHOUT threshold-based pruning
# =========================================================

def evaluate_individual_no_pruning(
    individual, evaluation_model, X_test, y_test
):
    # Important: no thresholding is applied to the weights.
    individual_array = np.asarray(individual, dtype=np.float32)

    try:
        set_weights(evaluation_model, individual_array)
    except Exception as e:
        print(f"Error setting model weights: {e}")
        return 0.0,

    y_pred_proba = evaluation_model.predict(
        X_test, verbose=0
    )
    y_pred = np.argmax(y_pred_proba, axis=1)

    accuracy = accuracy_score(y_test, y_pred)

    # Preserve the original class-weighted F1 calculation.
    class_weights = {
        0: 3.0,
        1: 2.0,
        2: 1.0,
        3: 1.0,
        4: 1.0
    }

    class_f1 = f1_score(
        y_test,
        y_pred,
        labels=[0, 1, 2, 3, 4],
        average=None,
        zero_division=0
    )

    weighted_f1 = sum(
        class_weights[i] * class_f1[i]
        for i in range(5)
    ) / sum(class_weights.values())

    # Keep the original fitness formula.
    # Without thresholding, size reflects exact non-zero weights.
    size = np.count_nonzero(individual_array) / len(individual_array)

    fitness = (
        0.5 * accuracy
        + 0.4 * weighted_f1
        - 0.1 * size
    )

    return fitness,


# =========================================================
# 6. Genetic algorithm WITHOUT pruning
# =========================================================

def run_genetic_algorithm_no_pruning(
    local_models, X_test, y_test
):
    print("\nStarting GA without threshold-based pruning")
    print(f"Population size: {POP_SIZE}")
    print(f"Generations: {GENERATIONS}")

    toolbox = base.Toolbox()

    toolbox.register(
        "individual",
        init_individual,
        local_models=local_models
    )

    toolbox.register(
        "population",
        tools.initRepeat,
        list,
        toolbox.individual
    )

    toolbox.register("mate", tools.cxOnePoint)

    toolbox.register(
        "mutate",
        tools.mutGaussian,
        mu=0.0,
        sigma=MUTATION_SIGMA,
        indpb=MUTATION_INDPB
    )

    toolbox.register(
        "select",
        tools.selTournament,
        tournsize=3
    )

    toolbox.register(
        "evaluate",
        evaluate_individual_no_pruning
    )

    evaluation_model = create_local_model()
    population = toolbox.population(n=POP_SIZE)

    hall_of_fame = tools.HallOfFame(1)
    accuracies_per_gen = []

    for gen in range(GENERATIONS):

        # Selection
        population = toolbox.select(
            population, len(population)
        )

        offspring = [
            toolbox.clone(ind) for ind in population
        ]

        # Crossover
        for child1, child2 in zip(
            offspring[::2], offspring[1::2]
        ):
            if random.random() < CROSSOVER_PROB:
                toolbox.mate(child1, child2)

                del child1.fitness.values
                del child2.fitness.values

        # Mutation
        for mutant in offspring:
            if random.random() < MUTATION_APPLY_PROB:
                toolbox.mutate(mutant)

                del mutant.fitness.values

        # Preserve the replacement strategy in the original code.
        if gen % 5 == 0 and gen > 0:
            new_population = toolbox.population(
                n=POP_SIZE // 2
            )
            offspring = (
                new_population
                + offspring[:POP_SIZE // 2]
            )

        # Evaluate individuals whose fitness is invalid.
        invalid_individuals = [
            ind for ind in offspring
            if not ind.fitness.valid
        ]

        fitnesses = [
            toolbox.evaluate(
                ind, evaluation_model, X_test, y_test
            )
            for ind in invalid_individuals
        ]

        for ind, fit in zip(
            invalid_individuals, fitnesses
        ):
            ind.fitness.values = fit

        population = offspring
        hall_of_fame.update(population)

        # Record the best individual's accuracy WITHOUT pruning.
        best_individual = hall_of_fame[0]
        best_weights = np.asarray(
            best_individual, dtype=np.float32
        )

        set_weights(evaluation_model, best_weights)

        y_pred = np.argmax(
            evaluation_model.predict(X_test, verbose=0),
            axis=1
        )

        accuracy = accuracy_score(y_test, y_pred)
        accuracies_per_gen.append(accuracy)

        print(
            f"Generation {gen + 1:02d}/{GENERATIONS} | "
            f"Accuracy: {accuracy:.4f} | "
            f"Fitness: {best_individual.fitness.values[0]:.4f}"
        )

    # =====================================================
    # 7. Final model: no thresholding
    # =====================================================

    best_individual = hall_of_fame[0]
    best_weights = np.asarray(
        best_individual, dtype=np.float32
    )

    set_weights(evaluation_model, best_weights)

    # Final predictions
    y_pred = np.argmax(
        evaluation_model.predict(X_test, verbose=0),
        axis=1
    )

    accuracy = accuracy_score(y_test, y_pred)

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    class_weights = {
        0: 3.0,
        1: 2.0,
        2: 1.0,
        3: 1.0,
        4: 1.0
    }

    class_f1 = f1_score(
        y_test,
        y_pred,
        labels=[0, 1, 2, 3, 4],
        average=None,
        zero_division=0
    )

    weighted_f1 = sum(
        class_weights[i] * class_f1[i]
        for i in range(5)
    ) / sum(class_weights.values())

    total_params = len(best_weights)
    nonzero_params = int(np.count_nonzero(best_weights))
    sparsity = 100 * (
        1 - nonzero_params / total_params
    )

    print("\n========== FINAL GA WITHOUT PRUNING ==========")
    print(f"Accuracy:           {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Macro F1:           {macro_f1:.4f}")
    print(f"Class-weighted F1:   {weighted_f1:.4f}")
    print(f"Total parameters:   {total_params}")
    print(f"Non-zero parameters:{nonzero_params}")
    print(f"Exact-zero sparsity:{sparsity:.2f}%")
    print("==============================================")

    # Save under NEW filenames; existing results remain unchanged.
    model_path = os.path.join(
        BASE_PATH,
        "global_model_GA_without_pruning_5class_timeseries168.weights.h5"
    )

    accuracy_path = os.path.join(
        BASE_PATH,
        "accuracies_per_gen_GA_without_pruning168.npy"
    )

    metrics_path = os.path.join(
        BASE_PATH,
        "metrics_GA_without_pruning168.npz"
    )

    evaluation_model.save_weights(model_path)

    np.save(accuracy_path, np.asarray(accuracies_per_gen))

    np.savez(
        metrics_path,
        accuracy=accuracy,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        total_params=total_params,
        nonzero_params=nonzero_params,
        sparsity_percent=sparsity
    )

    print("\nSaved model:", model_path)
    print("Saved generation accuracies:", accuracy_path)
    print("Saved final metrics:", metrics_path)

    return evaluation_model


# =========================================================
# 8. Load existing test data and local models
#    No preprocessing or local retraining is required.
# =========================================================

def main():
    X_test = np.load(
        os.path.join(
            BASE_PATH,
            "X_test_5class_timeseries168-1.npy"
        )
    )

    y_test = np.load(
        os.path.join(
            BASE_PATH,
            "y_test_5class_timeseries168-1.npy"
        )
    )

    local_models = []

    for i in range(5):
        model = create_local_model()

        weights_path = os.path.join(
            BASE_PATH,
            f"local_model_{i}_5class_timeseries168-1.weights.h5"
        )

        model.load_weights(weights_path)
        local_models.append(model)

    print("Loaded test set:", X_test.shape, y_test.shape)
    print("Loaded local models:", len(local_models))

    return run_genetic_algorithm_no_pruning(
        local_models,
        X_test,
        y_test
    )


global_model_no_pruning = main()
