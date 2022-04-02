import random
import numpy as np

from tetrisClasses import Board, Piece, Move, TetrisPlacementState
from tetrisUtilities import get_all_drop_moves, get_all_drop_boards, is_state_legal, is_state_goal, generate_boards_from_pieces
from myLogger import getModuleLogger


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
