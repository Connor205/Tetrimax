from queue import Queue
import tqdm

from tetrisClasses import Board, Piece, Move, TetrisPlacementState
from piece import Piece, Rotation
from constants import BOARD_WIDTH, BOARD_HEIGHT, BLOCK_CHARACTER


def get_all_legal_moves(
    b: Board,
    piece: Piece,
) -> list:
    """
    Given a board and a piece, return all legal moves
    """
    # So what we want to do here is generate all legal moves given a board and a piece
    # This will be accomplished by performing a BFS and returning all legal moves
    # Moves in this situation are simply (x, y, r) where x and y are the coordinates of the piece and r is the final rotation index

    # States are stored in dataclasses (we don't need to store the piece since it is constant accross the entire bfs)
    # We will need to be able to check if a given state is legal
    # We will also need to be able to check if a given state is a goal state
    # In order to check if a state is legal  we can check to see if any of the current pieces blocks are off the screen or if they intersect with any of the pieces on the board
    moves = set()
    visited = set()
    queue = Queue()
    # So we get to cheat here, when we generate all legal moves from a state, we can start only 4 above the highest block
    # This cuts out a huge portion of the bfs that is basically useless. Starting 4 above => subtacting 5
    # We always start at rotation 0, but allow the piece to be rotated to any valid position
    queue.put(
        TetrisPlacementState(b, BOARD_WIDTH // 2 - 2,
                             max(b.get_highest_block() - 5, 0), 0))
    while not queue.empty():
        state: TetrisPlacementState = queue.get()
        # Check to see if we have already been here
        if state in visited:
            continue
        visited.add(state)
        # Check to see if the state is legal (ie no overlapping or off board squares)
        if not is_state_legal(state, piece):
            continue  # If the state is not legal, we can skip it
        # Checking to see if we should add this to the list of possible moves
        if is_state_goal(state, piece):
            moves.add((state.x, state.y, state.rotation))
        # Here we deal with all of the rotations of the piece
        for rotationIndex in range(4):
            if state.rotation != rotationIndex:
                new_state = TetrisPlacementState(state.board, state.x, state.y,
                                                 rotationIndex)
                queue.put(new_state)
        queue.put(
            TetrisPlacementState(state.board, state.x + 1, state.y,
                                 state.rotation))  # Move right
        queue.put(
            TetrisPlacementState(state.board, state.x - 1, state.y,
                                 state.rotation))  # Move Left
        queue.put(
            TetrisPlacementState(state.board, state.x, state.y + 1,
                                 state.rotation))  # Move down
    return moves


def get_all_drop_moves(b: Board, piece: Piece):
    """
    This function is very similar to the above one, but rather than performing a bfs to find all of the more unique moves it simply drops the pieces in every single orientation from every legal position. 
    This generates all legal moves that would involve rotating, moving and the dropping the piece
    """
    moves = set()
    for i in range(4):
        rot = piece.get_rotation(i)
        left, right = rot.get_width_range()
        for x in range(left, right, 1):
            y = max(0, b.get_highest_block() - 5)  # init value for y
            while True:
                state = TetrisPlacementState(b, x, y, i)
                #TODO: Do some math to see if we really need to check legality here
                if not is_state_legal(state, piece):
                    break
                if is_state_goal(state, piece):
                    moves.add(Move(state.x, state.y, state.rotation))
                    break
                y += 1

    return moves


def get_all_drop_boards(b: Board, piece: Piece):
    moves = get_all_drop_moves(b, piece)
    boards = set()
    for move in moves:
        boards.add(b.make_move(piece, move)[0])
    return boards


def is_state_legal(state: TetrisPlacementState, piece):
    board = state.board
    x = state.x
    y = state.y
    rotation = state.rotation
    rot: Rotation = piece.get_rotation(rotation)
    # Here we simply want to check if any of the blocks are off screen x and y can be off the screen
    for off in range(4):
        if off + x < 0 or off + x >= BOARD_WIDTH:
            if any(rot.get_column(off)):
                return False
        if off + y >= BOARD_HEIGHT:
            if any(rot.get_row(off)):
                return False
    # Here we check to see if any of the blocks are overlapping
    for xOff in range(4):
        for yOff in range(4):
            if rot.get_pos(xOff, yOff):
                if board.get_square_truthy(x + xOff, y + yOff):
                    return False
    return True


def is_state_goal(state, piece):
    board = state.board
    x = state.x
    y = state.y
    rotation = state.rotation
    rot: Rotation = piece.get_rotation(rotation)
    for xOff in range(4):
        for yOff in range(4):
            if rot.get_pos(xOff, yOff) and board.get_square_truthy(
                    x + xOff, y + yOff + 1):
                return True
            if rot.get_pos(xOff, yOff) and y + yOff + 1 == BOARD_HEIGHT:
                return True
    return False


def generate_boards_from_pieces(pieces: list) -> list:
    """
    Given a list of pieces, generate all possible boards
    """
    print(f"Generating boards from {len(pieces)} pieces")
    print(f"{[p.name for p in pieces]}")
    losingBoards = []
    boards = [Board()]  # Here we are starting with an empty board
    for piece in pieces:
        print(
            f"Generating boards from {piece.name}, {len(boards)} boards so far"
        )
        newBoards = []
        for b in tqdm(boards):
            b: Board
            moves = get_all_legal_moves(b, piece, pullFromDict=True, save=True)
            for m in moves:
                newBoards.append(b.make_move(piece, m[2], m[0], m[1])[0])
            if len(moves) == 0:
                losingBoards.append(b)
        boards = list(set(newBoards))

    return boards, losingBoards