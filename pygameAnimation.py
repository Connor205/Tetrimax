import pygame
import pygame as pg
import pickle
import neat
from pygame.locals import QUIT, KEYDOWN, K_ESCAPE, K_LEFT, K_RIGHT, K_UP, K_DOWN

from myLogger import getModuleLogger
from tetrisAgent import NeatAgent
from tetrisSimulation import TetrisSimulation
from hueristics import featureVector
from constants import BOARD_HEIGHT, BOARD_WIDTH
from tetrisClasses import Board, Piece, Move, TetrisPlacementState

colors = {
    1: "lightBlue",
    2: 'darkBlue',
    3: 'orange',
    4: 'yellow',
    5: 'green',
    6: 'purple',
    7: 'red',
    0: 'white'
}


def getBoardsFromNeatAgent(genomePickleFile: str):
    # Load Genome from file using pickle
    with open(genomePickleFile, "rb") as f:
        genome = pickle.load(f)
    # Create network from genome
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         "tetrisagentconfig")
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    logger = getModuleLogger(__name__)
    logger.info("Testing neat agent")
    agent = NeatAgent(featureVector, net)
    sim = TetrisSimulation(agent)
    # Play a game from simulation
    board, score, survived, boards, moves, pieces, linesCleared = sim.playGame(
        scoringByLines=False)
    logger.info(f"Score: {score}")
    logger.info(f"Lines cleared: {linesCleared}")
    logger.info(f"Survived: {survived}")
    return boards, moves, pieces


def playAnimation(boards, moves, pieces, delay=.1):
    pg.init()
    SCREENRECT = pg.Rect(0, 0, 640, 480)
    screen = pg.display.set_mode(SCREENRECT.size)
    pg.display.set_caption("Tetris Game")
    clock = pg.time.Clock()
    frames = []
    prevBoard = Board()
    for b, m, p in zip(boards, moves, pieces):
        for i in range(0, m.y + 1):
            frames.append(prevBoard.make_move(p, Move(m.x, i, m.rotation))[0])
        prevBoard = b

    # Run our main loop whilst the player is alive.
    index = 0
    c = True
    while c:
        # get input
        for event in pg.event.get():
            if event.type == pg.QUIT:
                c = False
            if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                c = False
        b: Board = frames[index]
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                val = b.get_square(x, y)
                pg.draw.rect(screen, colors[val],
                             pg.Rect(x * 20 + 50, y * 20 + 50, 20, 20))
        # draw the screen
        pg.display.flip()
        pygame.time.delay(int(delay * 1000))
        index += 1
    pg.quit()


def main():
    boards, moves, pieces = getBoardsFromNeatAgent("neat-agent-95.pkl")
    print(len(boards))
    playAnimation(boards, moves, pieces)


if __name__ == "__main__":
    main()
