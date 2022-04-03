import logging
from typing import Tuple
from logging import getLogger

from tetrisClasses import Board, Piece, Move, TetrisPlacementState
from tetrisUtilities import get_all_drop_moves, get_all_drop_boards, is_state_legal, is_state_goal, generate_boards_from_pieces
from piece import Piece, Rotation, PIECES
from tetrisAgent import DepthAgent, NetworkAgent, TetrisAgent, SimpleAgent
from tetrisPieceGenerator import TetrisPieceGenerator
from hueristics import aggregateHeightHueristic, maxHeightHueristic, customHueristic, featureVector, originalFeatureVector
from myLogger import getModuleLogger


class TetrisSimulation:
    """
    Class for representing an entire tetris game
    Goal: Simulate a game given an agent and export a series of moves
    """

    def __init__(self, agent: TetrisAgent, numKnownPieces=3) -> None:
        self.logger = getModuleLogger(__name__, logging.DEBUG)
        if numKnownPieces < 1:
            raise ValueError("Must have at least one known piece")
        self.board = Board()
        self.score = 0
        self.game_over = False
        self.agent = agent
        self.numKnownPieces = numKnownPieces
        self.pieceGenerator = TetrisPieceGenerator()
        self.knownPieces = [
            next(self.pieceGenerator) for _ in range(numKnownPieces)
        ]
        self.logger.debug(f"Created TetrisSimulation")

    def playGame(self, scoringByLines=True) -> Tuple[Board, int]:
        """
        Simulates a single game of tetris based on the given agent, and returns the final score
        Currently score is calculated as the number of rows cleared
        """
        self.logger.debug(f"Simulating a game of Tetris")
        # Make sure that we are playing on an empty board
        if (self.board.get_board_sum() != 0):
            self.board = Board()
        assert (self.board.get_board_sum() == 0)  # Makes sure board is empty
        self.score = 0  # Reset score
        self.game_over = False  # Reset game over
        self.board = Board()  # Reset board
        numMoves = 0

        while not self.game_over and numMoves < 300:
            self.logger.debug("\n" + str(self.board))
            self.logger.debug("Bumpiness: " + str(self.board.get_bumpiness()))
            self.logger.debug("Holes: " + str(self.board.get_num_holes()))
            move = self.agent.get_move(self.board, self.knownPieces)
            # Here we catch the error for the case where the agent is unable to generate a move
            if move is None:
                self.game_over = True
                break

            # We update the board and grab the number of rows cleared
            self.board, rowsCleared = self.board.make_move(
                self.knownPieces[0], move)
            del move
            # Then we get rid of the pice we just played
            self.knownPieces.pop(0)
            # Add the next piece
            self.knownPieces.append(next(self.pieceGenerator))
            # And then add to the score
            if scoringByLines:
                self.score += rowsCleared
            else:
                self.score += rowsCleared * rowsCleared
            # This we check if the game is over
            if self.isGameOver():
                self.game_over = True
            numMoves += 1

        # Return the final board and score
        return self.board, self.score, self.game_over

    def isGameOver(self):
        """
        Checks to see if the game is over, we can use the bfs to generate all legal moves to see if the game is over,
        but that is super inefficient and not really accurate. 
        What we can do is we can check if any of the top row blocks are filled
        """
        return self.board.get_highest_block() == 0


def main():
    logger = getModuleLogger(__name__)
    simpleAgent = SimpleAgent()
    depthAgent = DepthAgent(customHueristic, depth=1)
    nnAgent = NetworkAgent(
        featureVectorGenerator=originalFeatureVector,
        weights=[
            0.5522740911902301, 0.5603935855519115, 0.8166220477829969,
            -0.0007715417257193602, 0.10041889348322228, -0.06,
            0.36860865859811925, 0.8743324936917122, 0.82, 0.646749612851632,
            1.3969439580921212, 0.25840916601216307, -0.07, 1.311174816954482,
            0.7479131995253523, 0.09, 0.33, 0.6675994685459831, 0.03, 0.1, 0.7,
            0.31416367031869624, 0.07562698517382904, 0.6812192749526834,
            0.13408758699227266, -0.36347999224470984, -0.1838375564764773,
            -0.591819910319467, 1.1373797927292544, -0.09, -0.44,
            -0.7547116450289864, -0.45, -0.04, -0.04, 0.16280420013336303,
            -0.77, 0.47107932108138073, 0.8929201934908048,
            -0.17314775047958286, 1.0139196495737148, -0.6968487833141848,
            -0.39, 0.4369063957067907, 0.6307541501166545, 0.926281732337451,
            -1.2368541578330263, 0.04679086384739073, -0.20192199867655267,
            1.4431845958042144, 0.23487987144636444, -1.2020875036072838, 0.43,
            -0.3017679642136074, 0.6592134177831279
        ])
    sim = TetrisSimulation(nnAgent)
    for i in range(3):
        board, score = sim.playGame()
        logger.info(f"Game {i} over, score: {score}")
        logger.info(f"Board: {board}")


if __name__ == "__main__":
    main()