# chessmc
MCTS based chess engine

Most chess engines use alpha-beta search algorithm, but chessmc uses Monte Carlo Tree Search (MCTS) algorithm.<br>
In the past decade or so, MCTS really has been growing to be much more popular than alpha-beta.

chessmc should be similar to AlphaZero in the way that it does not use random rollouts.<br>
Deep learning is used to evaluate board position. Trained net takes in a serialized board and outputs range from -1 (black wins) to 1 (white wins).

## Models
1. MLP (layers: 1048, 500, 50, epochs: 100, batch_size: 128) - tanh loss: 0.148981 

## Demo
![chessmc](https://user-images.githubusercontent.com/54076398/123994421-a7b34980-d9cd-11eb-8ef9-7e2174e5c09f.png)

## Dependencies
```
1. python-chess
2. numpy
3. pytorch
```

## Usage
```
1. python main.py  # runs web server on localhost:5000
2. Web browse to localhost:5000
```

## References
- trying this atm https://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/ICPRAM_CHESS_DNN_2018.pdf
