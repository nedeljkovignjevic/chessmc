import copy
import random

import chess.engine
import traceback

from chessmc.state import State
from chessmc.mcts import GameState, uct_search
from chessmc.utils import to_svg

from flask import Flask, request


app = Flask(__name__)


STATE = State()
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

        computer_move = uct_search(GameState(state=copy.deepcopy(STATE)), n_simulations=50)
        if chess.Move.from_uci(str(computer_move) + 'q') in STATE.board.legal_moves:
            computer_move.promotion = chess.QUEEN

        STATE.board.push_san(STATE.board.san(computer_move))
        if STATE.board.is_game_over():
            return app.response_class(response="Game over!", status=200)

    except Exception:
        traceback.print_exc()

    return app.response_class(response=STATE.board.fen(), status=200)


if __name__ == '__main__':
    app.run(debug=True)
