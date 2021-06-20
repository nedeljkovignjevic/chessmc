import collections
import numpy as np


class Node:
    def __init__(self, state, move, parent=None):
        self.state = state
        self.move = move
        self.parent = parent
        self.expanded = False
        self.children = {}
        self.children_priors = np.zeros([362], dtype=np.float32)
        self.children_total_values = np.zeros([362], dtype=np.float32)
        self.children_number_visits = np.zeros([362], dtype=np.float32)

    @property
    def number_visits(self):
        return self.parent.children_number_visits[self.move]

    @number_visits.setter
    def number_visits(self, value):
        self.parent.children_number_visits[self.move] = value

    @property
    def total_value(self):
        return self.parent.children_total_values[self.move]

    @total_value.setter
    def total_value(self, value):
        self.parent.children_total_values[self.move] = value

    def children_q(self):
        return self.children_total_values / (1 + self.children_number_visits)

    def children_u(self):
        return self.children_priors * np.sqrt(self.number_visits / (1.0 + self.children_number_visits))

    def best_child(self):
        return np.argmax(self.children_q() + self.children_u())

    def select_leaf(self):
        current = self
        while current.expanded:
            best_move = current.best_child()
            current = current.maybe_add_child(best_move)
        return current

    def expand(self, children_priors):
        self.expanded = True
        self.children_priors = children_priors

    def maybe_add_child(self, move):
        if move not in self.children:
            self.children[move] = Node(self.state.play(move), move, parent=self)
        return self.children[move]

    def backup(self, value_estimate: float):
        current = self
        while current.parent is not None:
            current.number_visits += 1
            current.total_value += (value_estimate * self.state.turn)
            current = current.parent


class TestNode(object):
    def __init__(self):
        self.parent = None
        self.children_total_values = collections.defaultdict(float)
        self.children_number_visits = collections.defaultdict(float)


class NeuralNet:
    @classmethod
    def evaluate(self, state):
        return 2.0, 1.0


def uct_search(state, n_simulations):
    root = Node(state, move=None, parent=TestNode())
    for _ in range(n_simulations):
        leaf = root.select_leaf()
        children_priors, value_estimate = NeuralNet.evaluate(leaf.state)
        leaf.expand(children_priors)
        leaf.backup(value_estimate)

    return np.argmax(root.children_number_visits)


class GameState():
    def __init__(self, turn=1):
        self.turn = turn

    def play(self, move):
        return GameState(-self.turn)


n_simulations = 10000
import time
tick = time.time()
uct_search(GameState(), n_simulations)
tock = time.time()
print("Took %s sec to run %s times" % (tock - tick, n_simulations))