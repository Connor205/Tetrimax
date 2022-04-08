from __future__ import print_function
from tetrisAgent import NeatAgent
from tetrisSimulation import TetrisSimulation
from hueristics import featureVector

import os
import neat
import math


def selu_activation(z):
    lam = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    return lam * z if z > 0.0 else lam * alpha * (math.exp(z) - 1)


def eval_single_genome(genome, config):
    total = 0
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    for _ in range(10):
        agent = NeatAgent(featureVector, net)
        sim = TetrisSimulation(agent)
        board, score, survived, boards, moves, pieces, linesCleared = sim.playGame(
            scoringType='tetris')
        total += score
    total /= 10
    # print("Finished Evaluating Genome")
    return total


def eval_genomes(genomes, config):
    evaluator = neat.ParallelEvaluator(16, eval_single_genome)
    evaluator.evaluate(genomes, config)


def run(config_file, checkpoint_file: str = None):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    config.genome_config.add_activation('my_selu', selu_activation)

    if checkpoint_file is not None:
        # Load the complete set of genomes that have been previously
        # evaluated.
        p = neat.Checkpointer.restore_checkpoint(checkpoint_file)
    else:
        # Create the population, which is the top-level object for a NEAT run.
        p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(
        neat.Checkpointer(5, filename_prefix='neat-checkpoints/selu-'))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 100)

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
    config_path = os.path.join(local_dir, 'tetrisagentconfigselu')
    run(config_path, 'neat-checkpoints/selu-4')