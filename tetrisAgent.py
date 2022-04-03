import random
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.python.keras.layers import Dense

from tetrisClasses import Board, Piece, Move, TetrisPlacementState
from tetrisUtilities import get_all_drop_moves, get_all_drop_boards, is_state_legal, is_state_goal, generate_boards_from_pieces
from myLogger import getModuleLogger
from hueristics import featureVector


class TetrisAgent():

    def __init__(self) -> None:
        self.logger = getModuleLogger(__name__)

    def get_move(self, board, pieces) -> Move:
        raise NotImplementedError()


class SimpleAgent(TetrisAgent):

    def get_all_moves(self, board: Board, piece: Piece) -> list:
        return get_all_drop_moves(board, piece)

    def get_move(self, board, pieces):
        moves = self.get_all_moves(board, pieces[0])
        return moves[0]


class RandomAgent(SimpleAgent):

    def get_move(self, board, pieces):
        moves = self.get_all_moves(board, pieces[0])
        return random.choice(list(moves))


class DepthAgent(SimpleAgent):

    def __init__(self, hueristic, depth=1) -> None:
        super().__init__()
        self.hueristic = hueristic  # here the hueristic is simply a function that takes in a board
        self.depth = depth

    def get_move(self, board, pieces):
        if len(pieces) < self.depth:
            raise ValueError(
                "Cannot perform depth search, not enough pieces provided")
        moves = self.get_all_moves(board, pieces[0])
        prevBoards = {}
        for m in moves:
            prevBoards[board.make_move(pieces[0], m)[0]] = m

        # Ok so here we are keeping track of the original move and the board that resulted from that move
        # We don't care about future moves, just the boards the represent
        # With the default depth value this is not run
        # If depth = 2 each move takes about .2 seconds, this is not fast enough
        for i in range(self.depth -
                       1):  # reducing depth since manually did first iteration
            newBoards = {}
            for b, move in prevBoards.items():
                possibleBoards = get_all_drop_boards(b, pieces[i + 1])
                for pb in possibleBoards:
                    newBoards[pb] = move
            prevBoards = newBoards
        boards = prevBoards
        self.logger.debug(f"Evaluating {len(boards)} boards")
        self.logger.debug(pieces[:self.depth])
        evaluations = [(self.hueristic(b), move) for b, move in boards.items()]
        if len(evaluations) == 0:
            return None
        return max(evaluations, key=lambda x: x[0])[1]


class FeatureAgent(SimpleAgent):
    """
    So lets be clear, this agent is AWESOME if weighted correctly. 
    Training this agent to infinitely clear lines is very doable
    """

    def __init__(self, featureVectorGenerator, weights):
        super().__init__()
        self.featureVectorGenerator = featureVectorGenerator
        self.weights = weights

    def get_move(self, board, pieces):
        moves = self.get_all_moves(board, pieces[0])
        prevBoards = {}
        for m in moves:
            prevBoards[board.make_move(pieces[0], m)[0]] = m
        evaluations = [(np.dot(self.featureVectorGenerator(b),
                               self.weights), move)
                       for b, move in prevBoards.items()]
        if len(evaluations) == 0:
            return None
        del prevBoards
        maxEval = max(evaluations, key=lambda x: x[0])[1]
        del evaluations
        return maxEval


class NetworkAgent(SimpleAgent):
    """So FeatureAgent is great, no issues BUT all of the behaviours are linear
    This is because thats all we need to win infinetley
    In theory we really dont need more than that. 
    BUT I WANNA WATCH AN AGENT SCORE A FKING TETRIS
    SOOOOO here we go. We can create an agent thats weights are actually the weights 
    for a CNN. Then we simply use the NN for evaluation of a given board state. 
    Noteably this agent will have the same inputs as the FeatureAgent, but will
    be allowed to have a more complex set of behaviors. 
    """

    def __init__(self, featureVectorGenerator, weights) -> None:
        super().__init__()
        self.featureVectorGenerator = featureVectorGenerator
        self.numFeatures = len(featureVectorGenerator(Board(), Board()))
        self.network = keras.Sequential([
            Dense(self.numFeatures,
                  activation="sigmoid",
                  name="layer1",
                  use_bias=False),
            Dense(self.numFeatures,
                  activation="relu",
                  name="layer2",
                  use_bias=False),
            Dense(1, name="layer3", use_bias=False)
        ])

        init_feature_vector = featureVectorGenerator(Board(), Board())
        fv = np.asmatrix(init_feature_vector)
        self.network(fv)
        composite_list = [
            np.array(weights[x:x + self.numFeatures])
            for x in range(0, len(weights), self.numFeatures)
        ]
        layer1Weights = np.array(composite_list[:self.numFeatures],
                                 dtype=object)
        layer2Weights = np.array(composite_list[self.numFeatures:2 *
                                                self.numFeatures],
                                 dtype=object)
        layer3Weights = np.asmatrix(
            np.array(composite_list[2 * self.numFeatures:], dtype=object))
        layer3Weights = layer3Weights.transpose()
        self.network.layers[0].set_weights([layer1Weights])
        self.network.layers[1].set_weights([layer2Weights])
        self.network.layers[2].set_weights([layer3Weights])
        # self.network.summary()

    def get_move(self, board, pieces):
        moves = self.get_all_moves(board, pieces[0])
        prevBoards = {}
        for m in moves:
            prevBoards[board.make_move(pieces[0], m)[0]] = m
        evaluations = [(self.network(
            np.asmatrix(self.featureVectorGenerator(b, board))).numpy()[0][0],
                        move) for b, move in prevBoards.items()]
        if len(evaluations) == 0:
            return None
        maxEval = max(evaluations, key=lambda x: x[0])[1]
        return maxEval


class MiniMaxAgent(TetrisAgent):

    def __init__(self, hueristic) -> None:
        super().__init__()
        self.hueristic = hueristic

    def get_move(self, board, pieces):
        return self.minimax(board, pieces, len(pieces))

    def minimax(self, board, pieces, depth) -> tuple:
        if depth == 0:
            return self.heuristic(board)
        best_move = None


def main():
    nnAgent = NetworkAgent(featureVector, np.zeros(55))


# Main Method
if __name__ == "__main__":
    main()