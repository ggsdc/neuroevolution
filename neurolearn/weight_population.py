import random
import time

from .weight_individual import IndividualWeights


class WeightPopulation:
    def __init__(
        self,
        population_size,
        max_generations,
        model,
        params,
        data,
        crossover_prob=0.25,
        mutation_prob=0.1,
    ):
        self.population_size = population_size
        self.max_generations = max_generations
        self.model = model
        self.data = data
        self.params = params
        self.weights = []
        self.mating_pool = []
        self.children = []
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob

        for i in range(self.population_size):
            self.weights.append(IndividualWeights(self.params))
            self.weights[i].create_random_individual()
        print("weights initialized")

    def evolve(self):
        print("time to evolve")
        for g in range(self.max_generations):
            self.calculate_fitness()
            self.select()
            self.crossover()
            self.substitution()

        self.get_pop()

    def calculate_fitness(self):

        for w in self.weights:
            w.calculate_fitness(self.model, self.data)

    def select(self):
        self.mating_pool = []
        for w in self.weights:
            self.mating_pool += [w for _ in range(round(w.validation_accuracy))]
            unit = random.uniform(0, 1)
            if unit < w.validation_accuracy % 1:
                self.mating_pool += [w]

    def crossover(self):
        self.children = []

        while len(self.children) < len(self.weights) / 2:

            first_parent = random.choice(self.mating_pool)
            second_parent = first_parent
            while first_parent.genes == second_parent.genes:
                second_parent = random.choice(self.mating_pool)

            first_child = IndividualWeights(self.params)
            second_child = IndividualWeights(self.params)

            for i in range(len(first_parent.genes)):
                interval = abs((first_parent.genes[i] - second_parent.genes[i]))
                if first_parent.genes[i] >= second_parent.genes[i]:
                    min_value = second_parent.genes[i] - self.crossover_prob * interval
                    max_value = first_parent.genes[i] + self.crossover_prob * interval
                else:
                    min_value = first_parent.genes[i] - self.crossover_prob * interval
                    max_value = second_parent.genes[i] + self.crossover_prob * interval

                if min_value < -2:
                    min_value = -2
                if max_value > 2:
                    max_value = 2

                first_child.genes.append(random.uniform(min_value, max_value))
                second_child.genes.append(random.uniform(min_value, max_value))

                # TODO: track parents
            first_child.mutate(self.mutation_prob)
            second_child.mutate(self.mutation_prob)

            first_child.calculate_fitness(self.model, self.data)
            second_child.calculate_fitness(self.model, self.data)

            self.children.append(first_child)
            self.children.append(second_child)

        self.weights += self.children

    def substitution(self):
        new_population = []
        old_population = [i for i in self.weights]

        while len(new_population) < self.population_size:
            selected = max(old_population, key=lambda i: i.validation_accuracy)
            old_population.remove(selected)
            new_population.append(selected)

        self.weights = new_population

    def get_pop(self):
        for w in self.weights:
            print(w.validation_accuracy)
