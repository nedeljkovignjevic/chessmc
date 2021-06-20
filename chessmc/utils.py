import base64
import chess.svg


def to_svg(state):
    return base64.b64encode(chess.svg.board(board=state.board).encode('utf-8')).decode('utf-8')