import random
import numpy as np
from tensorflow.keras.metrics import CategoricalAccuracy


class IndividualWeights:

    def __init__(self, size):
        self.size = size
        self.genes = []
        self.train_accuracy = 0
        self.validation_accuracy = 0
        self.trained = False

    def create_random_individual(self):
        for i in range(self.size):
            self.genes.append(random.normalvariate(0, 0.25))

    def calculate_fitness(self, model, data):
        if self.trained:
            return 0, 0
        x_train, y_train, x_validation, y_validation = data
        start = 0
        for i in range(len(model.layers)):
            if 'dropout' in model.layers[i].name:
                continue
            else:
                shape = (model.layers[i].input_shape[1], model.layers[i].units)
                end = start + shape[0] * shape[1]
                weights = np.array(self.genes[start:end]).reshape(shape)
                start = end
                end = end + shape[1]
                bias = np.array(self.genes[start:end]).reshape((shape[1],))
                start = end
                model.layers[i].set_weights([weights, bias])
        # self.train_accuracy = model.evaluate(x_train, y_train, verbose=0, batch_size=1024*8)[1]
        # self.validation_accuracy = model(x_validation, y_validation, verbose=0, batch_size=1024)
        metric = CategoricalAccuracy()
        metric.update_state(y_validation, model(x_validation, training=False))
        self.validation_accuracy = float(metric.result().numpy())

        self.trained = True

    def mutate(self, mutation_prob):
        while True:
            chance = random.uniform(0, 1)
            if chance <= mutation_prob:
                gene = random.randint(0, self.size - 1)
                new_value = random.normalvariate(0, 0.25)
                self.genes[gene] = new_value
                # TODO: track mutations
            else:
                break
