import numpy as np

from tetrisClasses import Board, Piece, Move, TetrisPlacementState


def maxHeightHueristic(board: Board):
    return board.get_highest_block()


def sumHueristic(board: Board):
    return board.get_board_sum()


def bumpinessHueristic(board: Board):
    return board.get_bumpiness()


def holesHueristic(board: Board):
    return board.get_num_holes()


def aggregateHeightHueristic(board: Board):
    return -board.get_aggregate_height()


def customHueristic(board: Board):
    return -0.798752914564018 * board.get_normalized_height(
    ) + -0.24921408023878 * bumpinessHueristic(
        board) + -0.164626498034284 * holesHueristic(board) - 99999 * (
            1 if board.is_lost() else 0)


def originalFeatureVector(board: Board):
    return np.array([
        board.get_normalized_height(),
        board.get_aggregate_height(),
        board.get_num_holes(),
        board.get_bumpiness(), 1 if board.is_lost() else 0
    ])


def featureVector(board: Board, prevBoard: Board):
    return np.array([
        board.get_normalized_height(),
        board.get_aggregate_height(),
        board.get_num_holes(),
        board.get_bumpiness(),
        board.get_num_row_transitions(),
        board.get_num_column_transitions(),
        board.get_num_pits(),
        (prevBoard.get_num_blocks() + 4 - board.get_num_blocks()) //
        10,  # Equivalent to number of lines cleared
    ])