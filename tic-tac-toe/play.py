from tkinter import Tk, Button
from tkinter.font import Font
from copy import deepcopy

import collections
import copy
import random

import numpy as np
import time

current_child = 0

class Board:
 
  def __init__(self,other=None):
    self.player = 'X'
    self.opponent = 'O'
    self.empty = '.'
    self.size = 3
    self.fields = {}
    for y in range(self.size):
      for x in range(self.size):
        self.fields[x,y] = self.empty
    # copy constructor
    if other:
      self.__dict__ = deepcopy(other.__dict__)
 
  def move(self,x,y):
    board = Board(self)
    board.fields[x,y] = board.player
    (board.player,board.opponent) = (board.opponent,board.player)
    return board
 
  def get_empty(self):
    ret = []
    for y in range(self.size):
        for x in range(self.size):
            if self.fields[x,y] == self.empty:
                ret.append((x,y))
    return ret

  def __minimax(self, player):
    if self.won():
      if player:
        return (-1,None)
      else:
        return (+1,None)
    elif self.tied():
      return (0,None)
    elif player:
      best = (-2,None)
      for x,y in self.fields:
        if self.fields[x,y]==self.empty:
          value = self.move(x,y).__minimax(not player)[0]
          if value>best[0]:
            best = (value,(x,y))
      return best
    else:
      best = (+2,None)
      for x,y in self.fields:
        if self.fields[x,y]==self.empty:
          value = self.move(x,y).__minimax(not player)[0]
          if value<best[0]:
            best = (value,(x,y))
      return best

  def uct_search(self, state, n_simulations):
    if self.won() or self.tied():
      return None
    root = Node(state, move=None, parent=TestNode())
    root.expanded = True
    visit_children(root)

    for _ in range(n_simulations):  # if state.children_len > 100 else state.children_len):
        leaf = root.select_leaf()
        children_priors, value_estimate = NeuralNet.evaluate(leaf.state)
        leaf.expand(children_priors)
        leaf.backup(value_estimate)
    return root.empty_fields[np.argmin(root.children_total_values)]
 
  def best_mm(self):
    return self.__minimax(True)[1]

  def best(self):
    return self.uct_search(GameState(board=copy.deepcopy(self)), n_simulations=2000)

  def best_rand(self):
    return random_move(self)
 
  def tied(self):
    for (x,y) in self.fields:
      if self.fields[x,y]==self.empty:
        return False
    return True
 
  def won(self, opponent='N'):
    # horizontal
    opp = self.opponent
    if opponent == 'O':
        opp = 'O'
    for y in range(self.size):
      winning = []
      for x in range(self.size):
        if self.fields[x,y] == opp:
          winning.append((x,y))
      if len(winning) == self.size:
        return winning
    # vertical
    for x in range(self.size):
      winning = []
      for y in range(self.size):
        if self.fields[x,y] == opp:
          winning.append((x,y))
      if len(winning) == self.size:
        return winning
    # diagonal
    winning = []
    for y in range(self.size):
      x = y
      if self.fields[x,y] == opp:
        winning.append((x,y))
    if len(winning) == self.size:
      return winning
    # other diagonal
    winning = []
    for y in range(self.size):
      x = self.size-1-y
      if self.fields[x,y] == opp:
        winning.append((x,y))
    if len(winning) == self.size:
      return winning
    # default
    return None
 
  def __str__(self):
    string = ''
    for y in range(self.size):
      for x in range(self.size):
        string+=self.fields[x,y]
      string+="\n"
    return string
 
class GUI:
 
  def __init__(self):
    self.app = Tk()
    self.app.title('TicTacToe')
    self.app.resizable(width=False, height=False)
    self.board = Board()
    self.font = Font(family="Helvetica", size=70)
    self.buttons = {}
    for x,y in self.board.fields:
      handler = lambda x=x,y=y: self.move(x,y)
      button = Button(self.app, command=handler, font=self.font, width=2, height=1)
      button.grid(row=y, column=x)
      self.buttons[x,y] = button
    handler = lambda: self.reset()
    simulate_minmax = lambda: self.simulate()
    simulate_rand = lambda: self.simulate_rand()
    simulate_rand_minmax = lambda: self.simulate_minmax_rand()
    button = Button(self.app, text='reset', command=handler)
    button.grid(row=self.board.size+1, column=0, columnspan=self.board.size, sticky="WE")
    button2 = Button(self.app, text='sim_min_max', command=simulate_minmax)
    button2.grid(row=self.board.size+2, column=0, columnspan=self.board.size, sticky="WE")
    button3 = Button(self.app, text='sim_random', command=simulate_rand)
    button3.grid(row=self.board.size+3, column=0, columnspan=self.board.size, sticky="WE")
    button4 = Button(self.app, text='sim_random_minmax', command=simulate_rand_minmax)
    button4.grid(row=self.board.size+4, column=0, columnspan=self.board.size, sticky="WE")
    self.update()
 
  def reset(self):
    self.board = Board()
    self.update()

  def simulate(self):
    won, lost, tied, i = 0, 0, 0, 0
    while i < 100:
      empties = self.board.get_empty()
      first = random.randint(0, len(empties)-1)
      first_move = empties[first]
      self.board = self.board.move(*first_move)
      self.update()
      while not self.board.won() and not self.board.tied():
        self.move_sim()
        #time.sleep(0.1)
        movemm = self.board.best_mm()
        if movemm:
          self.board = self.board.move(*movemm)
          self.update()
        #time.sleep(0.1)
      if self.board.tied():
        tied += 1
      else:
        if self.board.won('O'):
          won += 1
        else:
          lost += 1
      self.reset()
      i += 1
    print("MINMAX ---- Won: " + str(won) + ", Lost: " + str(lost) + ", Drawn: " + str(tied))

  def simulate_rand(self):
    won, lost, tied, i = 0, 0, 0, 0
    while i < 100:
      while not self.board.won() and not self.board.tied():
        rand_move = self.board.best_rand()
        self.board = self.board.move(*rand_move)
        self.update()
        self.move_sim()
        #time.sleep(0.1)
      if self.board.tied():
        tied += 1
      else:
        if self.board.won('O'):
          won += 1
        else:
          lost += 1
      self.reset()
      i += 1
    print("RANDOM ---- Won: " + str(won) + ", Lost: " + str(lost) + ", Drawn: " + str(tied))
  
  def simulate_minmax_rand(self):
    won, lost, tied, i = 0, 0, 0, 0
    while i < 100:
      while not self.board.won() and not self.board.tied():
        rand_move = self.board.best_rand()
        self.board = self.board.move(*rand_move)
        self.update()
        movemm = self.board.best_mm()
        if movemm:
          self.board = self.board.move(*movemm)
          self.update()
        #time.sleep(0.1)
      if self.board.tied():
        tied += 1
      else:
        if self.board.won('O'):
          won += 1
        else:
          lost += 1
      self.reset()
      i += 1
    print("RANDOM V MINMAX ---- Won: " + str(won) + ", Lost: " + str(lost) + ", Drawn: " + str(tied))

  def move_sim(self):
    move = self.board.best()
    if move:
      self.board = self.board.move(*move)
      self.update()
 
  def move(self,x,y):
    self.app.config(cursor="watch")
    self.app.update()
    self.board = self.board.move(x,y)
    self.update()
    move = self.board.best()
    if move:
      self.board = self.board.move(*move)
      self.update()
    self.app.config(cursor="")
 
  def update(self):
    for (x,y) in self.board.fields:
      text = self.board.fields[x,y]
      self.buttons[x,y]['text'] = text
      self.buttons[x,y]['disabledforeground'] = 'black'
      if text==self.board.empty:
        self.buttons[x,y]['state'] = 'normal'
      else:
        self.buttons[x,y]['state'] = 'disabled'
    winning = self.board.won()
    if winning:
      for x,y in winning:
        self.buttons[x,y]['disabledforeground'] = 'red'
      for x,y in self.buttons:
        self.buttons[x,y]['state'] = 'disabled'
    for (x,y) in self.board.fields:
      self.buttons[x,y].update()
 
  def mainloop(self):
    self.app.mainloop()

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
        self.empty_fields = state.board.get_empty()
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
                #continue
                return current
        return current

    def expand(self, children_priors):
        self.expanded = True
        # self.children_priors = children_priors * self.children_priors
        board = copy.deepcopy(self.state.board)
        while len(board.get_empty()) > 0:
            board = board.move(*random_move(board))
        is_win = board.won()
        if is_win:
            self.win = True
            self.total_value += 1
        self.number_visits += 1

    def maybe_add_child(self, move):
        if move >= len(self.children) or self.children[move] is None:
            self.children[move] = Node(self.state.play(move, self), move, parent=self)
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
           

def random_move(board):
    empties = board.get_empty()
    first = random.randint(0, len(empties)-1)
    return empties[first]
    #return random.choice([move for move in board.get_empty()])


class TestNode(object):
    def __init__(self):
        self.parent = None
        self.children_total_values = collections.defaultdict(float)
        self.children_number_visits = collections.defaultdict(float)


class NeuralNet:
    @classmethod
    def evaluate(self, state):
        return 2.0, 1.0


def visit_children(root):
    for i in range(root.state.children_len):
        root.children[i] = Node(root.state.play(i, root), i , parent=root)
        root.children[i].expanded = True
        board1 = copy.deepcopy(root.children[i].state.board)
        while len(board1.get_empty()) > 0:
            move = random_move(board1)
            board1 = board1.move(*move)

        is_win = board1.won()
        if is_win:
            root.children[i].win = True
            root.children[i].total_value += 1
        root.children[i].number_visits += 1
        root.children[i].backup(1)


class GameState:
    def __init__(self, turn=1, board=None):
        self.turn = turn
        self.board = board
        self.children_len = len(board.get_empty())

    def play(self, move, parent):
        board_ch = copy.deepcopy(self.board)
        x, y = parent.empty_fields[move]
        board = board_ch.move(x,y)
        return GameState(-self.turn, board)

 
if __name__ == '__main__':
  GUI().mainloop()