import pickle
import tqdm
from tetris import Board

SAVED_MOVES_NAME = "savedMoves.dat"
SAVED_BOARDS_NAME = "savedBoards.dat"

with open(SAVED_MOVES_NAME, "rb") as savedMovesFile:
    d = pickle.load(savedMovesFile)
boardDict = {}
with open(SAVED_BOARDS_NAME, "rb") as savedBoardsFile:
    for k, v in tqdm(d.items()):
        board, piece = k
        board: Board
        moves = v
        boardDict[(board, piece)] = [
            board.make_move(piece, m[2], m[0], m[1]) for m in moves
        ]
    pickle.dump(boardDict, savedBoardsFile)
