from torch.functional import Tensor

from chessmc.model import Model

import torch
import collections
import copy
import random

import numpy as np


model = Model()
model.load_state_dict(torch.load('models/mlp-stockfish-new.pth', map_location=torch.device('cpu'))['state_dict'])
model.eval()

current_child = 0


class Node:
    def __init__(self, state, move, parent=None):
        self.state = state
        self.move = move
        self.parent = parent
        self.expanded = False
        self.children = {}
        self.children_priors = np.zeros([self.state.children_len], dtype=np.float32)
        self.children_priors.fill(2)
        self.children_total_values = np.zeros([self.state.children_len], dtype=np.float32)
        self.children_number_visits = np.zeros([self.state.children_len], dtype=np.float32)
        self.win = False

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
            try:
                global current_child
                best_move = current.best_child()
                current_child = best_move
                current = current.maybe_add_child(best_move)
            except:
                continue
        return current

    def expand(self, children_priors):
        self.expanded = True
        # self.children_priors = children_priors * self.children_priors
        state = copy.deepcopy(self.state.state)
        while state.board.is_game_over() is not True:
            move = state.board.san(random_move(state))
            state.board.push_san(move)
        winner = False #state.board.outcome().winner
        result = state.board.result()
        if (result == '1-0' and state.board.turn) or (result == '0-1' and not state.board.turn) or result == '1/2-1/2':
            winner = True
        if winner is True:
            self.win = True
            self.total_value += 1
        self.number_visits += 1

    def maybe_add_child(self, move):
        if move >= len(self.children) or self.children[move] is None:
            self.children[move] = Node(self.state.play(move), move, parent=self)
        return self.children[move]

    def backup(self, value_estimate: float):
        current = self
        while current.parent.parent is not None:
            #wins += current.total_value
            #visits += current.number_visits
            current.parent.total_value = np.sum(current.parent.children_total_values, dtype=np.float32)
            if current.parent.win is True:
                current.parent.total_value += 1
            current.parent.number_visits = np.sum(current.parent.children_number_visits, dtype=np.float32) + 1
            current = current.parent
           

def random_move(state):
    return random.choice([move for move in state.legal_moves])


def net_evaluation_move(state):
    successors = []
    for move in state.legal_moves:
        state.board.push_san(str(move))
        successors.append(torch.argmax(model(Tensor(state.serialize()))))
        state.board.pop()

    if 0 in successors:
        return state.legal_moves[successors.index(0)]
    elif 2 in successors:
        return state.legal_moves[successors.index(2)]
    elif successors:
        return successors[0]
    else:
        return []


class TestNode(object):
    def __init__(self):
        self.parent = None
        self.children_total_values = collections.defaultdict(float)
        self.children_number_visits = collections.defaultdict(float)


class NeuralNet:
    @classmethod
    def evaluate(self, state):
        return 2.0, 1.0


def visit_children(root, to_range):
    for i in range(to_range):
        root.children[i] = Node(root.state.play(i), i, parent=root)
        root.children[i].expanded = True
        state1 = copy.deepcopy(root.children[i].state.state)
        while state1.board.is_game_over() is not True:
            move = state1.board.san(random_move(state1))
            state1.board.push_san(move)
        winner = False #state1.board.outcome().winner
        result = state1.board.result()
        if (result == '1-0' and state1.board.turn) or (result == '0-1' and not state1.board.turn) or result == '1/2-1/2':
            winner = True
        if winner is True:
            root.children[i].win = True
            root.children[i].total_value += 1
        root.children[i].number_visits += 1
        root.children[i].backup(1)


def uct_search(state, n_simulations):
    root = Node(state, move=None, parent=TestNode())
    root.expanded = True
    visit_children(root, root.state.children_len)

    for _ in range(n_simulations):  # if state.children_len > 100 else state.children_len):
        leaf = root.select_leaf()
        children_priors, value_estimate = NeuralNet.evaluate(leaf.state)
        leaf.expand(children_priors)
        leaf.backup(value_estimate)
    return state.state.legal_moves[np.argmax(root.children_total_values)]


class GameState:
    def __init__(self, turn=1, state=None):
        self.turn = turn
        self.state = state
        self.children_len = len(state.legal_moves) if state.legal_moves else 0

    def play(self, move):
        state = copy.deepcopy(self.state)
        move_str = self.state.board.san(self.state.legal_moves[move])
        state.board.push_san(move_str)
        return GameState(-self.turn, state)