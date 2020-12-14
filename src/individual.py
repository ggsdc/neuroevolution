import time

import pandas as pd
import tensorflow as tf
from tensorflow import keras

from src.gramatic import DerivationTree


class Individual:

    def __init__(self, layer_weight, architecture_weight):
        self.model = None
        self.fitness = None
        self.accuracy_train = 0
        self.accuracy_validation = 0
        self.mutation_prob = 0
        self.layers = 0
        self.neurons = 0
        self.structure = None
        self.derivation_tree = None
        self.depth = 0
        self.trained = False
        self.time = 0
        self.layer_weight = layer_weight
        self.architecture_weight = architecture_weight

    def random_individual(self):
        self.derivation_tree = DerivationTree()
        self.derivation_tree.create_tree()
        self.structure = self.derivation_tree.word
        self.neurons = self.structure.count('n') * 16
        self.layers = self.structure.count('/') - 1
        self.depth = self.derivation_tree.depth

    def create_child(self, parent1, parent2):
        pass

    def mutate(self):
        pass

    def train(self, x_train, y_train, x_val, y_val):
        if self.trained:
            return True
        n_epochs = 1024
        learning_rate = 0.001
        batch_size = 64

        self.model = keras.Sequential(name='DeepFeedForward')
        self.model.add(keras.layers.InputLayer(input_shape=(x_train.shape[1], ), batch_size=None))

        for x in self.structure.split('/'):
            if 'n' not in x:
                continue
            else:
                self.model.add(keras.layers.Dense(x.count('n') * 16, activation='relu', kernel_initializer='normal'))
                self.model.add(keras.layers.Dropout(0.3))

        self.model.add(keras.layers.Dense(y_train.shape[1], activation='softmax'))

        self.model.compile(loss=tf.keras.losses.categorical_crossentropy,
                           optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=["categorical_accuracy"])

        start = time.clock()
        history = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epochs, verbose=1,
                                 validation_data=(x_val, y_val))
        end = time.clock()
        self.time = (end - start)

        results = pd.DataFrame(history.history)
        self.accuracy_train = results.categorical_accuracy.values[-1:][0]
        self.accuracy_validation = results.val_categorical_accuracy.values[-1:][0]

        self.fitness = - self.architecture_weight * (
                    self.layer_weight * self.layers + (1 - self.layer_weight) * self.neurons) + (
                                   1 - self.architecture_weight) * self.accuracy_validation * 100

    def translate(self):
        pass
