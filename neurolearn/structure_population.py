from .structure_individual import IndividualStructure


class StructuresPopulation:

    def __init__(self, population_size, max_generations, mutation_prob, max_depth, method='genetic-algorithm', data=None, layer_weight=0, architecture_weight=0):
        self.population_size = population_size
        self.max_generations = max_generations
        self.method = method
        self.data = data
        self.population = []
        self.idx = 1
        for _ in range(self.population_size):
            self.population.append(
                IndividualStructure(self.idx, mutation_prob, max_depth, layer_weight, architecture_weight, 100))
            self.idx += 1

    def evolve(self):
        for g in range(self.max_generations):
            for i in self.population:
                i.train(self.method, self.data)

            break

    def get_population_structure(self):
        for i in self.population:
            print(i.structure)
