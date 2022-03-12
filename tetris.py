from re import I
from typing import Tuple


class Tetris:
    """
    Class for representing an entire tetris game
    Goal: Simulate a game given an agent and export a series of moves
    """

    def __init__(
        self,
        width: int = 10,
        height: int = 20,
    ) -> None:
        self.width = width
        self.height = height
        self.board = [[0 for _ in range(width)] for _ in range(height)]
        self.score = 0
        self.game_over = False
