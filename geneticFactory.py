import logging
import queue
import random
from tqdm import tqdm
from multiprocessing import Process, Queue
import time
import tensorflow as tf
from typing import List, Tuple

# Using a GPU is kinda unnecesssary here and threading is faster since we dont actually train any models
tf.config.set_visible_devices([], 'GPU')

from tetrisSimulation import TetrisSimulation
from myLogger import getModuleLogger
from tetrisAgent import FeatureAgent, NetworkAgent
from tetrisClasses import Board, Piece, Move, TetrisPlacementState
from hueristics import featureVector
import csv


class GeneticFactory:

    def __init__(self,
                 featureGenerator,
                 totalPopulation=1000,
                 linear=True,
                 numGames=5,
                 scoring='lines',
                 csvFile=None) -> None:
        self.logger = getModuleLogger(__name__, logging.DEBUG)
        b = Board()
        # We find the number of weights based on the number of features on an empty board
        self.featureGenerator = featureGenerator
        self.numFeatures = len(featureGenerator(b, b))
        self.linear = linear
        if linear:
            self.numWeights = self.numFeatures
        else:
            self.numWeights = self.numFeatures * self.numFeatures * 2 + self.numFeatures * 2
        self.logger.debug("Number of features: {}".format(self.numFeatures))
        self.logger.debug("Number of weights: {}".format(self.numWeights))
        self.totalPopulation = totalPopulation
        self.numGames = numGames
        self.scoring = scoring
        self.csvFile = csvFile
        # Checking to make sure that the given file is not there already
        if self.csvFile is not None:
            open(self.csvFile, 'x').close()
            with open(self.csvFile, "a", newline='') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow([
                    "Generation", "Max Score", "Total Score", "Average Score",
                    "Weights", "Time for Generation"
                ])

    def generatePopulation(self):

        def generateRandomWeights():
            return [
                round(random.uniform(-4, 4), 2) for _ in range(self.numWeights)
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
        # So this evaluator will constantly be checking for new jobs
        # and will evaluate them and put them back on the evaluation queue
        while True:
            if queue.empty():
                time.sleep(.5)
                continue
            weights = queue.get()
            # If it is passed false then it will exit
            if weights == False:
                break
            if self.linear:
                agent = FeatureAgent(self.featureGenerator, weights)
            else:
                agent = NetworkAgent(self.featureGenerator, weights)
            sim = TetrisSimulation(agent, numKnownPieces=1)
            games = [
                sim.playGame(scoringType=self.scoring)
                for _ in range(self.numGames)
            ]
            scores = []
            for game in games:
                board, score, survived, boards, moves, pieces, linesCleared = game
                scores.append(score)
            # Here we are passing back in the weight
            evaluationQueue.put((sum(scores) / len(scores), weights))
        print("This Process Is Finished, recieved false from queue")

    def runSimulation(self, generations=10):
        pop = self.generatePopulation()
        for i in range(generations):
            self.logger.info("Generation {}".format(i))
            evaluations = []
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
            self.logger.debug("Process {} started".format(j))

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
                self.logger.debug("Current Evaluation Length: {}".format(
                    len(evaluations)))
                # If we get to a point where our number of evaluations the same as our population, then we
                # can stop
                if len(evaluations) == self.totalPopulation:
                    break
            # Doing some logging
            self.logger.debug("Evaluations Length: {}".format(
                len(evaluations)))
            m = max(evaluations, key=lambda x: x[0])
            self.logger.info("Max Average Score: {}".format(m[0]))
            self.logger.info("Max Weights: {}".format(m[1]))
            endtime = time.time()
            self.logger.info("Time to run generation: {}".format(endtime -
                                                                 startTime))
            # Lets go ahead and log some of those rather important stats onto the csv file
            if self.csvFile is not None:
                with open(self.csvFile, "a", newline='') as csvFile:
                    writer = csv.writer(csvFile)
                    # lets go ahead and compile some stats about the current generation
                    scores = [x[0] for x in evaluations]
                    maxScore = max(scores)
                    totalScore = sum(scores)
                    avgScore = totalScore / len(scores)
                    mWeights = m[1]
                    t = endtime - startTime
                    writer.writerow(
                        [i, maxScore, totalScore, avgScore, mWeights, t])

            pop = self.computeNextGeneration(evaluations)

        # We want to kill all the processes
        for j in range(threads):
            q.put(False)
        for p in processList:
            p.join()
        self.logger.debug("All Threads Successfully Joined")
        # We want to sort the evaluations by score and return the best one
        return sorted(evaluations, key=lambda x: x[0])[-1]


def trainGeneticAgent(featureGenerator, totalPopulation=1000, generations=10):
    factory = GeneticFactory(featureGenerator, totalPopulation=totalPopulation)
    best_agent = factory.runThreadedSimulation(generations=generations,
                                               threads=16)
    print(best_agent)


def main():
    factory = GeneticFactory(featureVector,
                             totalPopulation=200,
                             linear=True,
                             scoring='tetris',
                             csvFile='linearTetris.csv')
    best_agent = factory.runThreadedSimulation(generations=5, threads=16)


# Main function
if __name__ == "__main__":
    main()
