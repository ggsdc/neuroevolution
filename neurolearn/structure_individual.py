import time

import pandas as pd
import tensorflow as tf
from tensorflow import keras

from .gramatic import DerivationTree
from .weight_population import WeightPopulation


class IndividualStructure:
    def __init__(
        self,
        idx,
        mutation_prob,
        max_depth,
        layer_weight,
        architecture_weight,
        max_generations=0,
    ):
        self.idx = idx
        self.model = None
        self.fitness = None
        self.accuracy_train = 0
        self.accuracy_validation = 0
        self.mutation_prob = mutation_prob
        self.max_depth = max_depth
        self.layers = 0
        self.neurons = 0
        self.structure = None
        self.derivation_tree = None
        self.depth = 0
        self.trained = False
        self.time = 0
        self.layer_weight = layer_weight
        self.architecture_weight = architecture_weight
        self.population = []
        self.max_generations = max_generations
        self.random_individual()

    def random_individual(self):
        self.derivation_tree = DerivationTree(self.max_depth)
        self.derivation_tree.create_tree()
        self.structure = self.derivation_tree.word
        self.neurons = self.structure.count("n") * 16
        self.layers = self.structure.count("/") - 1
        self.depth = self.derivation_tree.depth

    def create_child(self, parent1, parent2):
        pass

    def mutate(self):
        pass

    def train(self, method="genetic-algorithm", data=None):
        x_train, y_train, x_val, y_val = data
        if self.trained:
            return True
        print("Begin training ", self.idx)
        learning_rate = 0.001

        self.model = keras.Sequential(name="DeepFeedForward")
        self.model.add(
            keras.layers.InputLayer(input_shape=(x_train.shape[1],), batch_size=None)
        )

        for x in self.structure.split("/"):
            if "n" not in x:
                continue
            else:
                self.model.add(
                    keras.layers.Dense(
                        x.count("n") * 16,
                        activation="relu",
                        kernel_initializer="normal",
                    )
                )
                self.model.add(keras.layers.Dropout(0.3))

        self.model.add(keras.layers.Dense(y_train.shape[1], activation="softmax"))

        self.model.compile(
            loss=tf.keras.losses.categorical_crossentropy,
            optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
            metrics=["categorical_accuracy"],
        )

        if method == "back-propagation":
            self._train_gd(data)
        elif method == "genetic-algorithm":
            self._train_ag(data)

    def _train_ag(self, data):
        print("TRAIN")
        params = self.model.count_params()
        self.population = WeightPopulation(
            100, self.max_generations, self.model, params, data
        )
        self.population.evolve()

    def _train_gd(self, data):
        x_train, y_train, x_val, y_val = data
        n_epochs = 1024
        batch_size = 64
        start = time.clock()
        history = self.model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=n_epochs,
            verbose=0,
            validation_data=(x_val, y_val),
        )
        end = time.clock()
        self.time = end - start

        results = pd.DataFrame(history.history)
        self.accuracy_train = results.categorical_accuracy.values[-1:][0]
        self.accuracy_validation = results.val_categorical_accuracy.values[-1:][0]

        self.fitness = (
            -self.architecture_weight
            * (self.layer_weight * self.layers + (1 - self.layer_weight) * self.neurons)
            + (1 - self.architecture_weight) * self.accuracy_validation * 100
        )

        print(self.idx, " - ", self.fitness)
        self.trained = True

    def translate(self):
        pass
