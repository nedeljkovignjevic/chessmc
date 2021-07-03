import copy
import random
from copy import deepcopy

import chess.engine
import traceback

from chessmc.state import State
from chessmc.utils import to_svg

from flask import Flask, request

from chessmc.mcts import GameState, uct_search

app = Flask(__name__)


class MCTSState(State):
    # for lib mcts
    def getCurrentPlayer(self):
        return 1 if self.board.turn else -1

    def getPossibleActions(self):
        return self.legal_moves

    def takeAction(self, action):
        state = deepcopy(self)
        state.board.push_san(str(action))
        return state

    def isTerminal(self):
        return self.board.is_game_over()

    def getReward(self):
        result = self.board.result()
        if result == '1-0' and self.board.turn:
            return 1
        elif result == '0-1' and not self.board.turn:
            return 1
        else:
            return 0

    def __eq__(self, other):
        return self == other


STATE = MCTSState()

engine = chess.engine.SimpleEngine.popen_uci("stockfish_14_win_x64_avx2/stockfish_14_x64_avx2")


def random_move(state):
    return random.choice([move for move in state.legal_moves])


@app.route("/")
def home():
    return open("static/index.html").read().replace('start', STATE.board.fen())


@app.route("/new-game")
def new_game():
    STATE.board.reset()
    return app.response_class(response=STATE.board.fen(), status=200)


@app.route("/self-play")
def self_play():
    state = State()
    ret = '<html><head>'

    while not state.board.is_game_over():
        move = state.board.san(random_move(state))
        state.board.push_san(move)
        ret += '<img width=600 height=600 src="data:image/svg+xml;base64,%s"></img><br/>' % to_svg(state)

    print(state.board.result())
    return ret
        

@app.route("/move")
def move():
    if STATE.board.is_game_over():
        return app.response_class(response="Game over!", status=200)

    source = int(request.args.get('from', default=''))
    target = int(request.args.get('to', default=''))
    promotion = True if request.args.get('promotion', default='') == 'true' else False

    move = STATE.board.san(chess.Move(source, target, promotion=chess.QUEEN if promotion else None))

    if move is None or move == '':
        return app.response_class(response=STATE.board.fen(), status=200)

    try:
        STATE.board.push_san(move)
        if STATE.board.is_game_over():
            return app.response_class(response="Game over!", status=200)

        # searcher = mcts(iterationLimit=50)
        # best_action = searcher.search(initialState=STATE)
        #
        # computer_move = STATE.board.san(best_action)
        # STATE.board.push_san(computer_move)

        # result = engine.play(STATE.board, chess.engine.Limit(time=0.1))
        # stockfish_move = result.move
        computer_move = uct_search(GameState(state=copy.deepcopy(STATE)), n_simulations=200)
        if chess.Move.from_uci(str(computer_move) + 'q') in STATE.board.legal_moves:
            # promote to queen
            computer_move.promotion = chess.QUEEN
        STATE.board.push_san(STATE.board.san(computer_move))
        if STATE.board.is_game_over():
            return app.response_class(response="Game over!", status=200)

    except Exception:
        traceback.print_exc()

    return app.response_class(response=STATE.board.fen(), status=200)


if __name__ == '__main__':
    app.run(debug=True)
