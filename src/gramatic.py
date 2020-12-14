AXIOM = 'S'
NON_TERMINAL = ['S', 'A', 'H', 'Z', 'N']
TERMINAL = ['n', '/', 'I', 'O']

PRODUCTION_RULES = {'S': ['AH/Z'], 'A': ['I'], 'Z': ['O'], 'H': ['HH', '/N'], 'N': ['nN', 'n']}

import random


class Node:

    def __init__(self, symbol, parent, depth, idx):
        self.symbol = symbol
        self.parent = parent
        self.children = []
        self.depth = depth
        if self.parent is not None:
            self.idx = self.parent.idx + idx
        else:
            self.idx = idx
        if symbol in NON_TERMINAL:
            self.type = 0
        else:
            self.type = 1

    def add_children(self, children):
        self.children = children

    def children_word(self):
        word = ''
        for child in self.children:
            if len(child.children) > 0:
                word += child.children_word()
            else:
                word += child.symbol

        return word

    def __str__(self):
        return str(self.symbol)

    def __repr__(self):
        return str(self.symbol)


class DerivationTree:

    def __init__(self):
        self.word = AXIOM
        self.depth_dict = {}
        self.depth = 1
        self.idx = (1,)

        self.depth_dict[self.depth] = [Node(AXIOM, None, self.depth, self.idx)]

    def create_tree(self):
        while True:
            self.depth += 1
            self.depth_dict[self.depth] = []
            stop = True
            for node in self.depth_dict[self.depth-1]:
                if node.symbol in NON_TERMINAL:
                    substitution = [Node(val, node, self.depth, (index + 1,)) for index, val in
                                                   enumerate(random.choice(PRODUCTION_RULES[node.symbol]))]

                    self.depth_dict[self.depth] += substitution

                    node.add_children(substitution)
                    stop = False

            if stop:
                self.depth_dict.pop(self.depth)
                self.depth -= 1

                break

        self.word = self.depth_dict[1][0].children_word()





