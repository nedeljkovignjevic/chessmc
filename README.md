# chessmc
MCTS based chess engine

Most chess engines use alpha-beta search algorithm, but chessmc uses Monte Carlo Tree Search (MCTS) algorithm. It is expected to play at level of amateur human (~1500 Elo).

chessmc should be similar to AlphaZero in the way that it does not use random rollouts. Deep learning should be used for move selection and position evaluation.

### Idea 1
- UCT search algorithm (UCB1 = U + Q)
- U = (neural net output) / (number of simulations)
- Q = c + sqrt( number of visits for parent / (1 + (number of visits for current node)) )
- Trained net takes in a serialized board and outputs range from -1 (black wins) to 1 (white wins)
