import numpy as np
from deap import base, creator, tools
import random
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf
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

def set_weights(model, weights):
    weights = np.array(weights)
    new_weights = []
    idx = 0
    for w in model.get_weights():
        size = w.size
        if idx + size > len(weights):
            raise ValueError(f"Input weights are too short.")
        new_weights.append(weights[idx:idx+size].reshape(w.shape))
        idx += size
    model.set_weights(new_weights)

def init_individual(local_models):
    model = random.choice(local_models)
    weights = get_weights(model)
    return creator.Individual(weights.tolist())

def evaluate_individual(individual, test_model, X_test, y_test):
    individual_array = np.array(individual)
    individual_array[np.abs(individual_array) < COMPRESSION_THRESHOLD] = 0
    try:
        set_weights(test_model, individual_array)
    except Exception as e:
        print(f"Error in setting weights: {e}")
        return 0.0,
    y_pred_proba = test_model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    class_weights = {0: 3.0, 1: 2.0, 2: 1.0, 3: 1.0, 4: 1.0}
    f1 = f1_score(y_test, y_pred, average=None)
    weighted_f1 = sum(class_weights[i] * f1[i] for i in range(len(f1))) / sum(class_weights.values())

    accuracy = accuracy_score(y_test, y_pred)
    size = np.count_nonzero(individual_array) / len(individual_array)

    fitness = 0.5 * accuracy + 0.4 * weighted_f1 - 0.1 * size
    return fitness,

def run_genetic_algorithm(local_models, X_test, y_test):
    print("\nStep 3: Running genetic algorithm (time series, 5 classes)")
    global COMPRESSION_THRESHOLD, POP_SIZE, GENERATIONS
    COMPRESSION_THRESHOLD = 0.05
    POP_SIZE = 20
    GENERATIONS = 25

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("individual", init_individual, local_models=local_models)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.01)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate_individual)

    test_model = create_local_model()
    pop = toolbox.population(n=POP_SIZE)

    # Add list to store accuracy per generation
    accuracies_per_gen = []

    hof = tools.HallOfFame(1)
    for gen in range(GENERATIONS):
        pop = toolbox.select(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in pop]
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.8:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        for mutant in offspring:
            if random.random() < 0.5:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        if gen % 5 == 0 and gen > 0:
            new_pop = toolbox.population(n=POP_SIZE // 2)
            offspring = new_pop + offspring[:POP_SIZE // 2]
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(lambda ind: toolbox.evaluate(ind, test_model, X_test, y_test), invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        pop = offspring
        hof.update(pop)

        # Calculate accuracy of the best individual in this generation
        best_individual = hof[0]
        best_individual_array = np.array(best_individual)
        best_individual_array[np.abs(best_individual_array) < COMPRESSION_THRESHOLD] = 0
        set_weights(test_model, best_individual_array)
        y_pred_proba = test_model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies_per_gen.append(accuracy)
        print(f"Generation {gen+1}: Accuracy = {accuracy:.4f}")

    # Save accuracies to a file
    np.save(os.path.join(BASE_PATH, 'accuracies_per_gen168.npy'), accuracies_per_gen)

    best_individual = hof[0]
    best_individual_array = np.array(best_individual)
    best_individual_array[np.abs(best_individual_array) < COMPRESSION_THRESHOLD] = 0
    set_weights(test_model, best_individual_array)
    test_model.save_weights(os.path.join(BASE_PATH, 'global_model_5class_timeseries168-12.weights.h5'))

    return test_model

if __name__ == "__main__":
    random.seed(42)
    try:
        X_test = np.load(os.path.join(BASE_PATH, 'X_test_5class_timeseries168-1.npy'))
        y_test = np.load(os.path.join(BASE_PATH, 'y_test_5class_timeseries168-1.npy'))
        local_models = []
        for i in range(5):
            model = create_local_model()
            model.load_weights(os.path.join(BASE_PATH, f'local_model_{i}_5class_timeseries168-1.weights.h5'))
            local_models.append(model)
        global_model = run_genetic_algorithm(local_models, X_test, y_test)
    except FileNotFoundError as e:
        print(f"Error in loading files: {e}")
        raise