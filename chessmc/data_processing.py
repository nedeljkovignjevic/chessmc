import os
import chess.pgn
import chess.engine
import numpy as np
from state import State


def get_training_data():
    inputs, outputs = [], []

    results = {'1/2-1/2': 0, '0-1': -1, '1-0': 1}
    result_counter = {'1/2-1/2': 0, '0-1': 0, '1-0': 0}

    sample_num = 0
    for file in os.listdir('../data'):

        if os.path.isdir(os.path.join('../data', file)):
            continue

        pgn = open(os.path.join('../data', file), encoding='ISO-8859-1')

        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break

            result = game.headers['Result']
            if result not in results:
                continue

            result_counter[result] += 1
            if result_counter[result] >= 5000:
                continue

            result = results[result]
            board = game.board()
            print(f'parsing game {sample_num}, got {len(inputs)} samples')
            for i, move in enumerate(game.mainline_moves()):
                board.push(move)
                serialized_board = State(board).serialize()
                inputs.append(serialized_board)
                outputs.append(result)

            sample_num += 1

    return np.array(inputs), np.array(outputs), result_counter


def get_stockfish_training_data():
    engine = chess.engine.SimpleEngine.popen_uci("../stockfish_14_win_x64_avx2/stockfish_14_x64_avx2")
    inputs, outputs = [], []

    sample_num = 0
    counter = 0
    for file in os.listdir('../data'):

        if os.path.isdir(os.path.join('../data', file)):
            continue

        pgn = open(os.path.join('../data', file), encoding='ISO-8859-1')

        while True:

            game = chess.pgn.read_game(pgn)
            if game is None:
                break

            if sample_num < 6225:
                sample_num += 1
                continue

            board = game.board()
            counter = len(inputs)
            print(f'parsing game {sample_num}, got {counter} samples')
            for i, move in enumerate(game.mainline_moves()):
                board.push(move)
                score = engine.analyse(board, chess.engine.Limit(time=0.01))['score']
                score = score.relative.score()
                if score is None:
                    continue

                serialized_board = State(board).serialize()
                inputs.append(serialized_board)
                outputs.append(abs(score / 100))

            sample_num += 1
            if counter >= 1_000_000:
                np.savez('../data/stockfish_processed1M.npz', inputs, outputs)
                return

    return np.array(inputs), np.array(outputs)


if __name__ == '__main__':
    get_stockfish_training_data()

