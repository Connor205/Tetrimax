import pickle

SAVED_BOARDS_NAME = "savedBoards.dat"


class SavingMoves:

    def __init__(self):
        try:
            with open(SAVED_BOARDS_NAME, "rb") as savedBoardsFile:
                self.d = pickle.load(savedBoardsFile)
        except FileNotFoundError:
            self.d = {}
        self.currentNumberSaved = len(self.d)
        self.timesQueried = 0
        self.timesCachedUsed = 0

        print("Successfully Opened Saved Boards File With {} Entries".format(
            len(self.d)))

    def loadMoves(self, board, piece):
        self.timesQueried += 1
        if (board, piece) in self.d:
            self.timesCachedUsed += 1
            return self.d[(board, piece)]

    def saveBoards(self, board, piece, boards):
        self.d[(board, piece)] = boards

    def recordAllMoves(self):
        with open(SAVED_BOARDS_NAME, "wb") as savedBoardsFile:
            pickle.dump(self.d, savedBoardsFile)
        # print number of saved moves
        print("Successfully Saved {} move sets".format(
            len(self.d) - self.currentNumberSaved))
        self.currentNumberSaved = len(self.d)

    def printStats(self):
        print("Times Queried: {}".format(self.timesQueried))
        print("Times Cached Used: {}".format(self.timesCachedUsed))
        print("Times Cached Missed: {}".format(self.timesQueried -
                                               self.timesCachedUsed))
