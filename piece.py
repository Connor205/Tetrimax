# So how do we want to define a tetris piece
# Well there are 7 different pieces
# Each piece has 4 different rotations
# Each one has a name and color
# We can define each rotation as a different 4x4 matrix of booleans
# Each piece has a different way of rotating (this is important because they don't all rotate around the same point)
# Because i'm not fucking crazy I am going to use a python dataclass for this

from dataclasses import dataclass
from functools import cache
import random
from constants import BOARD_WIDTH


@dataclass(frozen=True)
class Rotation:
    matrix: tuple  # At its base value, a rotation is a 4x4 matrix of booleans in order to make it hashable we are using tuples

    # Here we are dealing with the requirements for a rotation
    def __post_init__(self):
        if len(self.matrix) != 4:
            raise ValueError("Rotation is not 4 elements tall")
        blocks = 0
        for i, row in enumerate(self.matrix):
            if len(row) != 4:
                raise ValueError(f"Row {i} is not 4 elements wide")
            blocks += sum(row)  # This also checks typing
        if blocks != 4:
            raise ValueError("Rotation does not have 4 blocks")

    def get_row(self, row: int) -> tuple:
        return self.matrix[row]

    def get_column(self, column: int) -> tuple:
        return tuple(row[column] for row in self.matrix)

    def get_pos(self, x: int, y: int) -> bool:
        return self.matrix[y][x]

    # These two functions could probably be stored somewhere so that they do not need to be recalculated every time
    # Not sure if @cache solves this
    @cache
    def get_furthest_left(self) -> int:
        for col in range(4):
            if any(self.get_column(col)):
                return col

    @cache
    def get_furthest_right(self) -> int:
        for col in range(3, -1, -1):
            if any(self.get_column(col)):
                return col

    @cache
    def get_width_range(self) -> tuple:
        return 0 - self.get_furthest_left(
        ), BOARD_WIDTH - self.get_furthest_right()


# Pieces need to be hashable, this means they cannot govern their own rotation
@dataclass(frozen=True)
class Piece:
    name: str
    color: str
    rotations: tuple  # we store exactly 4 rotations for every shape
    number: int

    #TODO: Right now we are forcing all pieces to have 4 rotations, this is not particularly optimal
    def __post__init__(self) -> None:
        # make sure there are exactly 4 rotations
        if len(self.rotations) != 4:
            raise ValueError(
                f"Piece ({self.name}) must have exactly 4 rotations")

    def get_rotation(self, num) -> Rotation:
        if num < 0 or num > 3:
            raise ValueError("Rotation number must be between 0 and 3")
        return self.rotations[num]

    def get_color(self) -> str:
        return self.color

    def get_name(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name


T = True
F = False

lineM = (
    Rotation(((F, F, F, F), (T, T, T, T), (F, F, F, F), (F, F, F, F))),
    Rotation(((F, F, T, F), (F, F, T, F), (F, F, T, F), (F, F, T, F))),
    Rotation(((F, F, F, F), (F, F, F, F), (T, T, T, T), (F, F, F, F))),
    Rotation(((F, T, F, F), (F, T, F, F), (F, T, F, F), (F, T, F, F))),
)

upperLeftM = (
    Rotation(((T, F, F, F), (T, T, T, F), (F, F, F, F), (F, F, F, F))),
    Rotation(((F, T, T, F), (F, T, F, F), (F, T, F, F), (F, F, F, F))),
    Rotation(((F, F, F, F), (T, T, T, F), (F, F, T, F), (F, F, F, F))),
    Rotation(((F, T, F, F), (F, T, F, F), (T, T, F, F), (F, F, F, F))),
)

upperRightM = (
    Rotation(((F, F, T, F), (T, T, T, F), (F, F, F, F), (F, F, F, F))),
    Rotation(((F, T, F, F), (F, T, F, F), (F, T, T, F), (F, F, F, F))),
    Rotation(((F, F, F, F), (T, T, T, F), (T, F, F, F), (F, F, F, F))),
    Rotation(((T, T, F, F), (F, T, F, F), (F, T, F, F), (F, F, F, F))),
)

squareM = (
    Rotation(((F, T, T, F), (F, T, T, F), (F, F, F, F), (F, F, F, F))),
    Rotation(((F, T, T, F), (F, T, T, F), (F, F, F, F), (F, F, F, F))),
    Rotation(((F, T, T, F), (F, T, T, F), (F, F, F, F), (F, F, F, F))),
    Rotation(((F, T, T, F), (F, T, T, F), (F, F, F, F), (F, F, F, F))),
)

zRightM = (
    Rotation(((F, T, T, F), (T, T, F, F), (F, F, F, F), (F, F, F, F))),
    Rotation(((F, T, F, F), (F, T, T, F), (F, F, T, F), (F, F, F, F))),
    Rotation(((F, F, F, F), (F, T, T, F), (T, T, F, F), (F, F, F, F))),
    Rotation(((T, F, F, F), (T, T, F, F), (F, T, F, F), (F, F, F, F))),
)

zLeftM = (
    Rotation(((T, T, F, F), (F, T, T, F), (F, F, F, F), (F, F, F, F))),
    Rotation(((F, F, T, F), (F, T, T, F), (F, T, F, F), (F, F, F, F))),
    Rotation(((F, F, F, F), (T, T, F, F), (F, T, T, F), (F, F, F, F))),
    Rotation(((F, T, F, F), (T, T, F, F), (T, F, F, F), (F, F, F, F))),
)

tM = (
    Rotation(((F, T, F, F), (T, T, T, F), (F, F, F, F), (F, F, F, F))),
    Rotation(((F, T, F, F), (F, T, T, F), (F, T, F, F), (F, F, F, F))),
    Rotation(((F, F, F, F), (T, T, T, F), (F, T, F, F), (F, F, F, F))),
    Rotation(((F, T, F, F), (T, T, F, F), (F, T, F, F), (F, F, F, F))),
)

orangeRicky = Piece("Orange Ricky", "orange", upperRightM, 3)
blueRicky = Piece("Blue Ricky", "darkBlue", upperLeftM, 2)
hero = Piece("Hero", "lightBlue", lineM, 1)
smashboy = Piece("Smashboy", "yellow", squareM, 4)
clevelandZ = Piece("Cleveland Z", "red", zLeftM, 7)
rhodeIslandZ = Piece("Rhode Island Z", "green", zRightM, 5)
teeWee = Piece("Tee Wee", "purple", tM, 6)
PIECES = set([
    orangeRicky,
    blueRicky,
    hero,
    smashboy,
    clevelandZ,
    rhodeIslandZ,
    teeWee,
])


def get_random_piece() -> Piece:
    return random.choice(list(PIECES))


def main():
    # This renders and displays all of the pieces
    import pygame as pg
    from pygame.locals import QUIT, KEYDOWN, K_ESCAPE, K_LEFT, K_RIGHT, K_UP, K_DOWN
    pg.init()
    SCREENRECT = pg.Rect(0, 0, 640, 480)
    screen = pg.display.set_mode(SCREENRECT.size)
    pg.display.set_caption("Tetris Pieces")
    clock = pg.time.Clock()

    # Run our main loop whilst the player is alive.
    c = True
    while c:
        # get input
        for event in pg.event.get():
            if event.type == pg.QUIT:
                c = False
            if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                c = False
        # draw the screen
        for i, piece in enumerate(pieces):
            currentY = i * 50 + 50
            for j in range(4):
                rotation = piece.get_rotation(j)
                currentX = j * 100 + 50
                pg.draw.rect(screen, "white",
                             pg.Rect(currentX, currentY, 40, 40))

                for k, row in enumerate(rotation.matrix):
                    yOff = k * 10
                    for l, col in enumerate(row):
                        xOff = l * 10
                        if col:
                            pg.draw.rect(
                                screen, piece.color,
                                pg.Rect(currentX + xOff, currentY + yOff, 10,
                                        10))
        pg.display.flip()
        clock.tick(40)

    pg.quit()


if __name__ == "__main__":
    main()
