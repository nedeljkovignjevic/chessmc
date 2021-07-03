import chess


class State(object):

    def __init__(self, board=None):
        self.board = chess.Board() if board is None else board

    @property
    def legal_moves(self):
        return list(self.board.legal_moves)

    def serialize(self):
        import numpy as np
        piece_dict = {"P": 1, "N": 1, "B": 1, "R": 1, "Q": 1, "K": 1,
                      "p": -1, "n": -1, "b": -1, "r": -1, "q": -1, "k": -1}

        state = np.zeros(768)
        idx = 0
        for k in piece_dict.keys():
            for i in range(64):
                piece = self.board.piece_at(i)
                if piece is not None:
                    turn = 1 if self.board.turn else -1
                    state[idx] = piece_dict[k] * turn if k == piece.symbol() else 0
                else:
                    state[idx] = 0

                idx += 1

        return state
