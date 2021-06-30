import os
import chess.pgn
import numpy as np

from state import State


def get_training_data():
    inputs, outputs = [], []

    results = {'1/2-1/2': 0, '0-1': -1, '1-0': 1}
    result_counter = {'1/2-1/2': 0, '0-1': 0, '1-0': 0}

    sample_num = 0
    for file in os.listdir('../content/data'):

        if os.path.isdir(os.path.join('../content/data', file)):
            continue

        pgn = open(os.path.join('../content/data', file), encoding='ISO-8859-1')

        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break

            result = game.headers['Result']
            if result not in results:
                continue

            result_counter[result] += 1

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


if __name__ == '__main__':
    x, y, results = get_training_data()
    print(f'Done. Results:\n 1/2 - {results["1/2-1/2"]}\n 0-1 - {results["0-1"]}\n 1-0 - {results["1-0"]}')
    np.savez('../data/processed.npz', x, y)
