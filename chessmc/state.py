import chess


class State(object):

    def __init__(self, board=None):
        self.board = chess.Board() if board is None else board

    @property
    def legal_moves(self):
        return list(self.board.legal_moves)

    def serialize(self):
        import numpy as np
        piece_dict = {"P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6,
                      "p": 9, "n": 10, "b": 11, "r": 12, "q": 13, "k": 14}

        state = np.zeros(64, np.uint8)
        for i in range(64):
            if (piece := self.board.piece_at(i)) is not None:
                state[i] = piece_dict[piece.symbol()]

        if self.board.has_queenside_castling_rights(chess.WHITE):
            assert state[0] == 4
            state[0] = 7
        if self.board.has_kingside_castling_rights(chess.WHITE):
            assert state[7] == 4
            state[7] = 7
        if self.board.has_queenside_castling_rights(chess.BLACK):
            assert state[56] == 4 + 8
            state[56] = 7 + 8
        if self.board.has_kingside_castling_rights(chess.BLACK):
            assert state[63] == 4 + 8
            state[63] = 7 + 8
        if self.board.ep_square is not None:
            assert state[self.board.ep_square] == 0
            state[self.board.ep_square] = 8

        state = state.reshape(8, 8)
        binary_state = np.zeros((5, 8, 8), np.uint8)
        binary_state[0] = (state >> 3) & 1
        binary_state[1] = (state >> 2) & 1
        binary_state[2] = (state >> 1) & 1
        binary_state[3] = (state >> 0) & 1
        binary_state[4] = (self.board.turn * 1.0)
        return binary_state
