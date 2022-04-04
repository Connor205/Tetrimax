from __future__ import annotations
from typing import Tuple

from constants import BOARD_WIDTH, BOARD_HEIGHT, BLOCK_CHARACTER
from dataclasses import dataclass
from piece import Piece, Rotation


@dataclass(frozen=True)
class Move:
    x: int
    y: int
    rotation: Rotation


class Board:

    def __init__(self, matrix: list = None) -> None:
        if matrix is None:
            self._matrix = tuple(
                tuple([0] * BOARD_WIDTH) for _ in range(BOARD_HEIGHT))
        else:
            h = len(matrix)
            w = len(matrix[0])
            if h != BOARD_HEIGHT or w != BOARD_WIDTH:
                raise ValueError(
                    f"Board must be {BOARD_WIDTH}x{BOARD_HEIGHT}, board is {w}x{h}"
                )
            self._matrix = tuple(tuple(row) for row in matrix)
        self._hashMatrix = tuple(
            [tuple([bool(x) for x in row]) for row in self._matrix])

    def get_square(self, x: int, y: int) -> int:
        if x < 0 or x >= BOARD_WIDTH or y < 0 or y >= BOARD_HEIGHT:
            return None
        return self._matrix[y][x]

    def get_square_truthy(self, x: int, y: int) -> bool:
        return self.get_square(x, y) is not None and self.get_square(x, y) > 0

    def get_row(self, row: int) -> tuple:
        return self._matrix[row]

    def get_column(self, col: int) -> tuple:
        return tuple(row[col] for row in self._matrix)

    def get_colmn_height(self, col: int) -> int:
        for i in range(BOARD_HEIGHT):
            if self.get_square_truthy(col, i):
                return i
        return BOARD_HEIGHT

    def get_normalized_height(self) -> int:
        return BOARD_HEIGHT - self.get_highest_block()

    def get_normalized_column_height(self, col: int) -> int:
        return BOARD_HEIGHT - self.get_colmn_height(col)

    def make_move(self,
                  piece: Piece,
                  move: Move,
                  scoringByLines=True) -> Tuple(Board, int):
        """
        Returns tuple of newBoard, lines cleared
        This function does not validate the move, will throw errors
        """
        newMatrix = list(list(row) for row in self._matrix)
        rot = piece.get_rotation(move.rotation)
        for xOff in range(4):
            for yOff in range(4):
                if rot.get_pos(xOff, yOff):
                    newMatrix[move.y + yOff][move.x + xOff] = piece.number
        # We need to remove cleared rows now
        linesToRemove = []
        for i in range(BOARD_HEIGHT):
            if all(newMatrix[i]):
                linesToRemove.append(i)
        for i in linesToRemove:
            newMatrix.pop(i)
            newMatrix.insert(0, [0] * BOARD_WIDTH)
        if scoringByLines:
            return Board(newMatrix), len(linesToRemove)
        else:  # Simple enough to reward
            return Board(newMatrix), len(linesToRemove) * len(linesToRemove)

    def get_board_sum(self):
        return sum(sum(row) for row in self._matrix)

    def get_highest_block(self) -> int:
        # Returns the row number of the highest block
        # Reminder that lower numbers are actually higher blocks since the 0,0 is upper left
        for i in range(BOARD_HEIGHT):
            if any(self.get_row(i)):
                return i
        return BOARD_HEIGHT

    def get_num_holes(self) -> int:
        total = 0
        for col in range(BOARD_WIDTH):
            c = self.get_column(col)
            found_block = False

            for i in range(BOARD_HEIGHT):
                if c[i] != 0:
                    found_block = True
                if found_block and c[i] == 0:
                    total += 1
        return total

    def get_bumpiness(self) -> int:
        total = 0
        for i in range(1, BOARD_WIDTH):
            total += abs(
                self.get_colmn_height(i) - self.get_colmn_height(i - 1))
        return total

    def get_aggregate_height(self) -> int:
        return sum(
            self.get_normalized_column_height(i) for i in range(BOARD_WIDTH))

    def is_lost(self) -> bool:
        return self.get_highest_block() == 0

    def get_num_pits(self) -> int:
        total = 0
        for i in range(BOARD_WIDTH):
            if self.get_normalized_column_height(i) == 0:
                total += 1
        return total

    def get_num_row_transitions(self):
        total = 0
        for i in range(BOARD_WIDTH):
            for j in range(BOARD_HEIGHT - 1):
                if self.get_square(i, j) != self.get_square(i, j + 1):
                    total += 1
        return total

    def get_num_column_transitions(self):
        total = 0
        for i in range(BOARD_HEIGHT - 1):
            for j in range(BOARD_WIDTH):
                if self.get_square(i, j) != self.get_square(i + 1, j):
                    total += 1
        return total

    def get_num_blocks(self) -> int:
        return sum(sum(row) for row in self._hashMatrix)

    def __repr__(self) -> str:

        def convertLineToString(line):
            return "".join([str(x) for x in line])
            return "".join([BLOCK_CHARACTER if x else "_" for x in line])

        return "\n".join(convertLineToString(line) for line in self._matrix)

    def __hash__(self) -> int:
        # Boards are the same if their matrix is the same based on booleans, not colors
        return hash(self._hashMatrix)

    def __eq__(self, other: Board) -> bool:
        return self._matrix == other._matrix


@dataclass(frozen=True)
class TetrisPlacementState:
    board: Board
    x: int
    y: int
    rotation: Rotation