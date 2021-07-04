# chessmc
MCTS based chess engine

Most chess engines use alpha-beta search algorithm, but chessmc uses Monte Carlo Tree Search (MCTS) algorithm.<br>
In the past decade or so, MCTS really has been growing to be much more popular than alpha-beta.

Chessmc should be similar to AlphaZero in the way that it does not use random rollouts.<br>
Deep learning is used to evaluate board position. Neural net is trained to reproduce Stockfish’s evaluation function.<br>
The output of the evaluation process is a value called the centipawn (cp). Centipawns correspond to  1/100th of a pawn and are the most commonly used  method when it comes to board evaluations.

Every board position has been labeled as Winning, Losing or Draw according to
the cp Stockfish assigns to it. <br>
A label of Winning has been assigned if cp > 1.5, Losing if it  was < −1.5 and Draw if the cp evaluation was between these 2 values.

## Data
https://www.pgnmentor.com/files.html

## Model
Neural Network architecture: MLP -> input: 768 | hidden: 256 | hidden: 64 | output: 3<br>
Leaky ReLU and Dropout 0.5 on both hidden layers<br>
Optimizer: Adam (with default params)<br>
Trained 100 epochs on 500_000 examples with 128 batch size<br>
Train loss: 0.08<br>
Test accuracy: 76%  

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
