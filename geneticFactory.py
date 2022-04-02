import logging
import random
from tqdm import tqdm

from tetrisSimulation import TetrisSimulation
from myLogger import getModuleLogger
from tetrisAgent import FeatureAgent
from tetrisClasses import Board, Piece, Move, TetrisPlacementState
from hueristics import featureVector


class GeneticFactory:

    def __init__(self, featureGenerator, totalPopulation=1000) -> None:
        self.logger = getModuleLogger(__name__, logging.DEBUG)
        b = Board()
        # We find the number of weights based on the number of features on an empty board
        self.featureGenerator = featureGenerator
        self.numWeights = len(featureGenerator(b))
        self.logger.debug("Number of weights: {}".format(self.numWeights))
        self.totalPopulation = totalPopulation

    def generatePopulation(self):

        def generateRandomWeights():
            return [
                round(random.uniform(-5, 5), 2) for _ in range(self.numWeights)
            ]

        return [generateRandomWeights() for _ in range(self.totalPopulation)]

    def computeNextGeneration(self, evaluations):
        self.logger.info("Computing next generation")
        # Evaluations are given as tuples of (score, weights)
        # We want to keep the top 30% of the population
        es = sorted(evaluations, key=lambda x: x[0], reverse=True)

        top = es[:len(es) // 3]

        totalFitness = sum([x[0] for x in top])
        self.logger.info(
            "Total Fitness Of Current Genertion: {}".format(totalFitness))

        if totalFitness != 0:
            probs = [x[0] / totalFitness for x in top]
        else:
            probs = [1 / len(top)] * len(top)
        # We want to randomly select the next generation
        nextGen = []
        for i in range(self.totalPopulation):
            if i < len(top):
                nextGen.append(top[i][1])
            else:
                # We want to select a random parent based on their fitness
                a, b = random.choices(top, k=2, weights=probs)
                a = a[1]
                b = b[1]
                newMans = []
                for i in range(self.numWeights):
                    choice = random.choice([a, b])
                    newMans.append(choice[i])
                nextGen.append(newMans)
        del top
        del es
        return nextGen

    def runSimulation(self, generations=10):
        pop = self.generatePopulation()
        for i in range(generations):
            self.logger.info("Generation {}".format(i))
            evaluations = []
            #TODO: Make this parallel, add in threading
            for j, weights in enumerate(pop):
                self.logger.debug("Generation: {}, agent: {}".format(i, j))
                agent = FeatureAgent(self.featureGenerator, weights)
                sim = TetrisSimulation(agent, numKnownPieces=1)
                games = [sim.playGame() for _ in range(3)]
                scores = [x[1] for x in games]
                evaluations.append((sum(scores) / 3, weights))
                del agent
                del sim
            m = max(evaluations, key=lambda x: x[0])
            self.logger.info("Max Average Score: {}".format(m[0]))
            self.logger.info("Max Weights: {}".format(m[1]))
            pop = self.computeNextGeneration(evaluations)
        return sorted(evaluations, key=lambda x: x[0])[-1]


def main():
    factory = GeneticFactory(featureVector, totalPopulation=100)
    pop = factory.runSimulation(generations=5)
    print()


# Main function
if __name__ == "__main__":
    main()
