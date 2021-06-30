# chessmc
MCTS based chess engine

Most chess engines use alpha-beta search algorithm, but chessmc uses Monte Carlo Tree Search (MCTS) algorithm.<br>
In the past decade or so, MCTS really has been growing to be much more popular than alpha-beta.

chessmc should be similar to AlphaZero in the way that it does not use random rollouts.<br>
Deep learning is used to evaluate board position. Trained net takes in a serialized board and outputs range from -1 (black wins) to 1 (white wins).

trying this atm https://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/ICPRAM_CHESS_DNN_2018.pdf