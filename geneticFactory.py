import logging
import queue
import random
from tqdm import tqdm
from multiprocessing import Process, Queue
import time
import tensorflow as tf
from typing import List

from tetrisSimulation import TetrisSimulation
from myLogger import getModuleLogger
from tetrisAgent import FeatureAgent, NetworkAgent
from tetrisClasses import Board, Piece, Move, TetrisPlacementState
from hueristics import featureVector


class GeneticFactory:

    def __init__(self, featureGenerator, totalPopulation=1000) -> None:
        self.logger = getModuleLogger(__name__, logging.INFO)
        b = Board()
        # We find the number of weights based on the number of features on an empty board
        self.featureGenerator = featureGenerator
        self.numFeatures = len(featureGenerator(b, b))
        self.numWeights = self.numFeatures * self.numFeatures * 2 + self.numFeatures
        self.logger.debug("Number of features: {}".format(self.numFeatures))
        self.totalPopulation = totalPopulation

    def generatePopulation(self):

        def generateRandomWeights():
            return [
                round(random.uniform(-1, 1), 2) for _ in range(self.numWeights)
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
                    val = choice[i]
                    # Here we have a 20% chance to mutate the value
                    if random.random() < 0.2:
                        val += random.gauss(0, .4)
                    newMans.append(val)
                nextGen.append(newMans)
        return nextGen

    def threadedEvaluator(self, queue: Queue, evaluationQueue: Queue):
        while True:
            if queue.empty():
                time.sleep(.5)
                continue
            weights = queue.get()
            if weights == False:
                break
            agent = NetworkAgent(self.featureGenerator, weights)
            sim = TetrisSimulation(agent, numKnownPieces=1)
            games = [sim.playGame(scoringByLines=False) for _ in range(3)]
            scores = [x[1] for x in games]
            evaluationQueue.put((sum(scores) / 3, weights))
        print("This Process Is Finished, recieved false from queue")

    def runSimulation(self, generations=10):
        pop = self.generatePopulation()
        for i in range(generations):
            self.logger.info("Generation {}".format(i))
            evaluations = []
            #TODO: Make this parallel, add in threading
            for j, weights in tqdm(enumerate(pop)):
                self.logger.debug("Generation: {}, agent: {}".format(i, j))
                agent = NetworkAgent(self.featureGenerator, weights)
                sim = TetrisSimulation(agent, numKnownPieces=1)
                games = [sim.playGame(scoringByLines=False) for _ in range(3)]
                scores = [x[1] for x in games]
                evaluations.append((sum(scores) / 3, weights))
            m = max(evaluations, key=lambda x: x[0])
            self.logger.info("Max Average Score: {}".format(m[0]))
            self.logger.info("Max Weights: {}".format(m[1]))
            pop = self.computeNextGeneration(evaluations)
        return sorted(evaluations, key=lambda x: x[0])[-1]

    def runThreadedSimulation(self, generations=10, threads=16):
        pop = self.generatePopulation()
        q = Queue()
        evaluationQueue = Queue()
        processList: List[Process] = []
        for j in range(threads):
            p = Process(target=self.threadedEvaluator,
                        args=(q, evaluationQueue))
            p.start()
            processList.append(p)
            self.logger.info("Process {} started".format(j))

        for i in range(generations):
            self.logger.info("Generation {}".format(i))
            startTime = time.time()
            # We are using a queue so that we can have multiple processes
            for weightVector in pop:
                q.put(weightVector)

            evaluations = []
            while True:
                if evaluationQueue.empty():
                    time.sleep(.5)
                    continue
                else:
                    evaluations.append(evaluationQueue.get())
                if len(evaluations) == self.totalPopulation:
                    break

            self.logger.debug("Evaluations Length: {}".format(
                len(evaluations)))
            m = max(evaluations, key=lambda x: x[0])
            self.logger.info("Max Average Score: {}".format(m[0]))
            self.logger.info("Max Weights: {}".format(m[1]))
            endtime = time.time()
            self.logger.info("Time to run generation: {}".format(endtime -
                                                                 startTime))
            pop = self.computeNextGeneration(evaluations)

        for j in range(threads):
            q.put(False)
        for p in processList:
            p.join()
        self.logger.debug("All Threads Successfully Joined")
        return sorted(evaluations, key=lambda x: x[0])[-1]


def trainGeneticAgent(featureGenerator, totalPopulation=1000, generations=10):
    factory = GeneticFactory(featureGenerator, totalPopulation=totalPopulation)
    best_agent = factory.runThreadedSimulation(generations=generations,
                                               threads=16)
    print(best_agent)


def main():
    trainGeneticAgent(featureVector, totalPopulation=1000, generations=50)


# Main function
if __name__ == "__main__":
    main()
