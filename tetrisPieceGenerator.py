from piece import Piece, Rotation, PIECES, get_random_piece, hero, smashboy, clevelandZ, teeWee, rhodeIslandZ, blueRicky
import random


class TetrisPieceGenerator:
    """
    Generator which manages the creation of tetris pieces as an infinite stream. 
    Notably follows tetris rules in that batches of 7 are shuffled and then all picked before they are reset.
    """

    def __init__(self) -> None:
        self.current = list(PIECES)
        random.shuffle(self.current)

    def __next__(self) -> Piece:
        if len(self.current) == 0:
            self.current = list(PIECES)
            random.shuffle(self.current)
        return self.current.pop()


#main method
if __name__ == "__main__":
    gen = TetrisPieceGenerator()
    for i in range(10):
        print(next(gen))