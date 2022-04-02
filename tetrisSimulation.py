import logging
from typing import Tuple
from logging import getLogger

from tetrisClasses import Board, Piece, Move, TetrisPlacementState
from tetrisUtilities import get_all_drop_moves, get_all_drop_boards, is_state_legal, is_state_goal, generate_boards_from_pieces
from piece import Piece, Rotation, PIECES
from tetrisAgent import DepthAgent, TetrisAgent, SimpleAgent
from tetrisPieceGenerator import TetrisPieceGenerator
from hueristics import aggregateHeightHueristic, maxHeightHueristic, customHueristic
from myLogger import getModuleLogger


class TetrisSimulation:
    """
    Class for representing an entire tetris game
    Goal: Simulate a game given an agent and export a series of moves
    """

    def __init__(self, agent: TetrisAgent, numKnownPieces=3) -> None:
        self.logger = getModuleLogger(__name__, logging.INFO)
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

    def playGame(self) -> Tuple[Board, int]:
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

        while not self.game_over and numMoves < 500:
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
            self.score += rowsCleared
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
    sim = TetrisSimulation(depthAgent)
    for i in range(3):
        board, score = sim.playGame()
        logger.info(f"Game {i} over, score: {score}")
        logger.info(f"Board: {board}")


if __name__ == "__main__":
    main()