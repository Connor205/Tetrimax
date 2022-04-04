import logging
from typing import Tuple, List
from logging import getLogger

from tetrisClasses import Board, Piece, Move, TetrisPlacementState
from tetrisUtilities import get_all_drop_moves, get_all_drop_boards, is_state_legal, is_state_goal, generate_boards_from_pieces
from piece import Piece, Rotation, PIECES
from tetrisAgent import DepthAgent, NetworkAgent, TetrisAgent, SimpleAgent
from tetrisPieceGenerator import TetrisPieceGenerator
from hueristics import aggregateHeightHueristic, maxHeightHueristic, customHueristic, featureVector, originalFeatureVector
from myLogger import getModuleLogger


class TetrisSimulation:
    """
    Class for representing an entire tetris game
    Goal: Simulate a game given an agent and export a series of moves
    """

    def __init__(self, agent: TetrisAgent, numKnownPieces=3) -> None:
        self.logger = getModuleLogger(__name__, logging.INFO)
        if numKnownPieces < 1:
            raise ValueError("Must have at least one known piece")
        self.board = Board()
        self.score = 0
        self.game_over = False
        self.agent = agent
        self.numKnownPieces = numKnownPieces
        self.pieceGenerator = TetrisPieceGenerator()
        self.knownPieces = [
            next(self.pieceGenerator) for _ in range(numKnownPieces)
        ]
        self.logger.debug(f"Created TetrisSimulation")

    def playGame(
        self,
        max_moves=300,
        scoringByLines=True
    ) -> Tuple[Board, int, bool, List[Board], List[Move], List[Piece], int]:
        """
        Simulates a single game of tetris based on the given agent, and returns the final score
        Currently score is calculated as the number of rows cleared
        """
        self.logger.debug(f"Simulating a game of Tetris")
        # Make sure that we are playing on an empty board
        if (self.board.get_board_sum() != 0):
            self.board = Board()
        assert (self.board.get_board_sum() == 0)  # Makes sure board is empty
        self.score = 0  # Reset score
        self.game_over = False  # Reset game over
        self.board = Board()  # Reset board
        numMoves = 0
        linesCleared = 0
        boards = []
        pieces = []
        moves = []

        while not self.game_over and numMoves < max_moves:
            self.logger.debug("-----------------------------")
            self.logger.debug("\n" + str(self.board))
            # self.logger.debug("Bumpiness: " + str(self.board.get_bumpiness()))
            # self.logger.debug("Holes: " + str(self.board.get_num_holes()))
            move = self.agent.get_move(self.board, self.knownPieces)
            # Here we catch the error for the case where the agent is unable to generate a move
            if move is None:
                self.game_over = True
                break

            # We update the board and grab the number of rows cleared
            self.board, rowsCleared = self.board.make_move(
                self.knownPieces[0], move)
            moves = moves + [move]
            boards = boards + [self.board]
            pieces = pieces + [self.knownPieces[0]]
            # Then we get rid of the pice we just played
            self.knownPieces.pop(0)
            # Add the next piece
            self.knownPieces.append(next(self.pieceGenerator))
            # And then add to the score
            if scoringByLines:
                self.score += rowsCleared
            else:
                self.score += rowsCleared * rowsCleared
            linesCleared += rowsCleared

            # This we check if the game is over
            if self.isGameOver():
                self.game_over = True
            numMoves += 1

        # Return the final board and score
        return self.board, self.score, not self.game_over, boards, moves, pieces, linesCleared

    def isGameOver(self):
        """
        Checks to see if the game is over, we can use the bfs to generate all legal moves to see if the game is over,
        but that is super inefficient and not really accurate. 
        What we can do is we can check if any of the top row blocks are filled
        """
        return self.board.get_highest_block() == 0


def main():
    logger = getModuleLogger(__name__)
    nnAgent = NetworkAgent(
        featureVectorGenerator=featureVector,
        weights=[
            0.34083502075858174, 1.3967296073240072, -2.101164907172171,
            0.5791912606685484, 0.8452356825334939, -2.9765637205900313,
            0.9842109661979721, 0.8931603972748186, 0.04045276380374113,
            -0.013757355087339218, -0.8652001004503095, -0.82908254441039,
            1.6413175309979935, 0.9784590122885507, 0.617540380615309,
            -1.438700620856368, 1.5570223464731494, 1.326511474917487,
            2.257369286525689, 0.6706809924783961, 0.9924417754676216,
            -0.14858873962304936, 2.2740983566609962, 0.2726127385674881,
            -1.9821556527965596, 1.6951031984254858, -0.6083733599893706,
            0.4120422714740943, 0.23869640611078108, 0.049565854936952314,
            -0.8095765201479718, -0.09675436805329451, -0.4969713765627104,
            -1.2251791954599818, 1.022841934316636, 0.19500246863121473,
            0.795069651371751, 0.7524059108507988, 0.09982736783368723,
            1.3753744939594221, -0.7506534161259167, -2.4945382900509006,
            -1.1249999164675741, 0.7411929172488867, 1.3551054991681366,
            -0.28385736975006254, 0.6319636551592772, -0.6229795884570585,
            0.9410945879323581, 0.5534063018002175, 0.045483079067163285,
            -2.805200250828465, 1.2385933984053115, -1.140810628263162,
            0.6618985615996854, 0.10050981565123207, 0.31631761895333677,
            0.7999058039629072, -0.9942785378896118, 0.20258278253800205,
            -1.6534470141482467, 0.43142530829984066, -0.8687352844102703,
            0.9537112615634487, 1.2708257663789513, 0.6069345098200484,
            -1.5652259482396091, 2.1184699000542806, 0.5009389434754019,
            0.7214199152640641, 0.6770642326776688, 1.9838399945647784,
            1.9804323694928494, 1.4183473383568836, -0.8347490217506855,
            2.2916714263351063, -0.06659752728162376, -0.7857491954538314,
            -1.1797211402208934, 0.31737057230545873, 1.0415166207745488,
            -0.1574491914097441, 0.6351212980371679, -0.20129218180440547,
            -0.9193239722589459, 0.7764482399879196, -0.37809521114993416,
            0.6969813468418712, 0.6937155443294365, -0.5096079264245595,
            -0.16434604343526127, 0.6591260792941762, 1.6879625153594018,
            0.47746425389955494, 0.6419926830458335, -0.2669915358110209,
            0.45156060281331156, 1.7090236199660964, 1.9942633665449143,
            -0.09506359623672754, -0.9606031590535022, 0.5090857653497048,
            1.0741411947840465, 1.3640921135788437, -1.0218030489573275,
            1.4303766252512566, -1.0406260147967763, -0.24890969340857524,
            -0.37761692827323934, 3.306453744177146, -1.807574235994301,
            1.3736914823038329, -1.4942051047275866, -2.430203868752634,
            -0.1843919707436789, -0.0371736419536664, -0.9975394008212807,
            1.2084962749540542, 1.2345083387537257, 0.47637196782986363,
            1.3540631577471323, 1.393463099814097, 0.9554659163670209,
            2.155622879126144, 0.31193407861241756, 0.22448605994497478,
            -1.381961929374024, -1.2316926732074303, 0.4601864346466168,
            -0.9290773766264945, 1.2580079306489291, -0.017081804259183653,
            0.30689478942213955, -1.4511825138728096, -0.1832319896546629,
            -0.9109153871329061, 1.2742849679804358, -0.6299343237410823,
            -0.8343922317085366, 1.9640410443111993, -1.312825864593824,
            -1.0363859970206022, -2.5940773745888825, -0.8597061197685395,
            -0.620654500287785, 0.49924812054096984, -0.10302892058533508,
            -2.1497557912706746, -1.0380228827808062, -0.002252252297482271,
            0.38270002051954993, -1.557091382422044, -1.9197836405714268,
            -0.03923899761826821, -1.4121834581824375, 0.29085989031577136,
            2.189148546018777, 0.5062388085566852, -1.6709295634449643,
            0.551546866525884, 0.48275758202755736, 0.7168382707738661,
            -0.6817839480001356, -0.805201903306244, 0.24456568413838609,
            -0.1062231594141265, -0.35219177417523917, -0.1299042041488106,
            1.5769542469437767, 0.06467875085501318, 0.615275340883685,
            1.0589638251285118, -0.8468368403101942, -2.1835923067272147,
            -0.8795363654270972, -2.5167788751408233, 2.5622806191034178,
            0.12579312631231834, -0.6167930443787186, -1.6242494892646155
        ])
    sim = TetrisSimulation(nnAgent)
    for i in range(10):
        board, score, survived, boards, moves, pieces, linesCleared = sim.playGame(
            scoringByLines=False)
        logger.info(f"Game {i} over, score: {score}")
        logger.info(f"Game {i} over, lines cleared: {linesCleared}")
        logger.info(f"Game {i} over, survived: {survived}")
        # logger.info(f"Board: {board}")


if __name__ == "__main__":
    main()