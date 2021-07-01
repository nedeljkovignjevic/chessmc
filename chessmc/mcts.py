import collections
import copy
import random

import numpy as np

current_child = 0


class Node:
    def __init__(self, state, move, parent=None):
        self.state = state
        self.move = move
        self.parent = parent
        self.expanded = False
        self.children = {}
        self.children_priors = np.zeros([self.state.children_len], dtype=np.float32)
        self.children_total_values = np.zeros([self.state.children_len], dtype=np.float32)
        self.children_number_visits = np.zeros([self.state.children_len], dtype=np.float32)
        self.win = 0
        self.lose = 0

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

    def children_q(self):   # average of all evaluations
        return self.children_total_values / (1 + self.children_number_visits)

    def children_u(self):   # as a node is repeatedly visited, bonus shrinks
        return self.children_priors * np.sqrt(self.number_visits / (1.0 + self.children_number_visits))

    def best_child(self):
        return np.argmax(self.children_q() + self.children_u())

    def select_leaf(self):
        current = self
        if current.parent.parent is None:
            current.expanded = True
        while current.expanded:
            # current.number_visits += 1     # fix for returning same variation each time
            # current.total_value -= 1    # fix for returning same variation each time
            best_move = current.best_child()
            global current_child
            current_child = best_move
            current = current.maybe_add_child(best_move)
        return current

    def expand(self, children_priors):
        self.expanded = True
        self.children_priors = children_priors * self.children_priors
        state = copy.deepcopy(self.state.state)
        while state.board.is_game_over() is not True:
            move = state.board.san(random_move(state))
            state.board.push_san(move)

        print(state.board)
        print(state.board.outcome())

        winner = state.board.outcome().winner
        if winner is True:
            self.win += 1
            # print("POBJEDAA")
        elif winner is False:
            self.lose += 1
            # print("PORAZ")

    def maybe_add_child(self, move):
        if move not in self.children:
            self.children[move] = Node(self.state.play(move), move, parent=self)
        return self.children[move]

    def backup(self, value_estimate: float, root):
        current = self
        wins = current.win
        visits = current.number_visits
        while current.parent is not None and current.parent is not root:
            # fix for returning same variation each time
            # current.number_visits += 1
            # current.total_value += value_estimate
            # current.total_value += value_estimate + 1
            current.total_value += (value_estimate * self.state.turn)
            # current.total_value += (value_estimate * self.state.turn) + 1
            current = current.parent


def random_move(state):
    return random.choice([move for move in state.legal_moves])


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

    for i in range(root.state.children_len):
        root.children[i] = Node(root.state.play(i), i, parent=root)
        root.children[i].expanded = True
        state = copy.deepcopy(root.children[i].state.state)
        while state.board.is_game_over() is not True:
            move = state.board.san(random_move(state))
            state.board.push_san(move)
        winner = state.board.outcome().winner
        if winner is True:
            root.children[i].win += 1
        elif winner is False:
            root.children[i].lose += 1
        root.children_number_visits[i] += 1
        root.children_total_values[i] += root.children[i].win

    for _ in range(n_simulations):  # if state.children_len > 100 else state.children_len):
        leaf = root.select_leaf()
        print(leaf.move)
        print(leaf.parent)
        print(leaf.children)
        children_priors, value_estimate = NeuralNet.evaluate(leaf.state)
        leaf.expand(children_priors)
        root.children_total_values[current_child] = leaf.win  # - leaf.lose
        root.children_number_visits[current_child] += 1
        # leaf.backup(value_estimate, root)
    print(root.children)
    print(root.children_number_visits)
    print(root.children_total_values)
    # return np.argmax(root.children_number_visits), np.argmax(root.children_priors), np.argmax(root.children_total_values)
    return state.state.legal_moves[np.argmax(root.children_total_values)]


class GameState:
    def __init__(self, turn=1, state=None):
        self.turn = turn
        self.state = state
        self.children_len = len(state.legal_moves) if state.legal_moves else 0

    def play(self, move):
        state = copy.deepcopy(self.state)
        # print(self.state.legal_moves)
        # print(move)
        move_str = self.state.board.san(self.state.legal_moves[move])
        state.board.push_san(move_str)
        return GameState(-self.turn, state)


# n_simulations = 10000
# import time
# tick = time.time()
# print(uct_search(GameState(), n_simulations))
# tock = time.time()
# print("Took %s sec to run %s times" % (tock - tick, n_simulations))