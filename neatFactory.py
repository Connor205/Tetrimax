from __future__ import print_function
from tetrisAgent import NeatAgent
from tetrisSimulation import TetrisSimulation
from hueristics import featureVector

import os
import neat


def eval_single_genome(genome, config):
    total = 0
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    for i in range(10):
        agent = NeatAgent(featureVector, net)
        sim = TetrisSimulation(agent)
        board, score, survived, boards, moves, pieces, linesCleared = sim.playGame(
            scoringByLines=False)
        total += score
    total /= 10
    # print("Finished Evaluating Genome")
    return total


def eval_genomes(genomes, config):
    evaluator = neat.ParallelEvaluator(16, eval_single_genome)
    evaluator.evaluate(genomes, config)


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 300)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print("Lets see how it does in 20 games")
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for i in range(20):
        agent = NeatAgent(winner_net)
        sim = TetrisSimulation(featureVector, agent)
        board, score, survived, boards, moves, pieces, linesCleared = sim.playGame(
            scoringByLines=False)
        print("Game {}: Score: {}".format(i, score))
        print("Game {}: Lines Cleared: {}".format(i, linesCleared))
        print("Game {}: Survived: {}".format(i, survived))


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'tetrisagentconfig')
    run(config_path)