import copy
import math
import random
import time
import random
from math import sqrt, log
from math import log, sqrt, inf
import numpy as np
from rich.table import Table
from rich.progress import track
from rich.console import Console
from rich.progress import Progress

import classes.logic as logic

# Base class for different player strategies in the game.
class PlayerStrat:
    def __init__(self, _board_state, player):
        """
        Initialize the player strategy with the current state of the board and the player number.

        :param _board_state: The current state of the board as a 2D list.
        :param player: The player number (1 or 2).
        """
        self.root_state = _board_state
        self.player = player

    def start(self):
        """
        Abstract method to select a tile from the board. To be implemented by subclasses.

        :returns: (x, y) tuple of integers corresponding to a valid and free tile on the board.
        """
        raise NotImplementedError

# Random strategy for a player. Chooses a move randomly from available tiles.
class RandomPlayer(PlayerStrat):
    def __init__(self, _board_state, player):
        super().__init__(_board_state, player)
        self.board_size = len(_board_state)

    def select_tile(self, board):
        """
        Randomly selects a free tile on the board.

        :param board: The current game board.
        :returns: (x, y) tuple of integers corresponding to a valid and free tile on the board.
        """
        free_tiles = [(x, y) for x in range(self.board_size) for y in range(self.board_size) if board[x][y] == 0]
        return random.choice(free_tiles) if free_tiles else None

    def start(self):
        return self.select_tile(self.root_state)

# MiniMax strategy for a player. Uses the MiniMax algorithm to choose the best move.



class MiniMax(PlayerStrat):
    def __init__(self, _board_state, player, depth=2):
        super().__init__(_board_state, player)
        self.board_size = len(_board_state)
        self.depth = depth

    def select_tile(self, board, player):
        best_score = float('-inf')
        best_move = None

        for x in range(self.board_size):
            for y in range(self.board_size):
                if board[x][y] == 0:
                    board[x][y] = player
                    score = self.minimax(board, self.depth - 1, False, player)
                    board[x][y] = 0
                    if score > best_score:
                        best_score = score
                        best_move = (x, y)

        return best_move

    def minimax(self, board, depth, is_maximizing, player, alpha=float('-inf'), beta=float('inf')):
        if depth == 0 or self.is_game_over(board):
            return self.evaluate_board(board, player)

        if is_maximizing:
            best_score = float('-inf')
            for x in range(self.board_size):
                for y in range(self.board_size):
                    if board[x][y] == 0:
                        board[x][y] = player
                        score = self.minimax(board, depth - 1, False, player, alpha, beta)
                        board[x][y] = 0
                        best_score = max(best_score, score)
                        alpha = max(alpha, best_score)
                        if beta <= alpha:
                            break
            return best_score
        else:
            best_score = float('inf')
            for x in range(self.board_size):
                for y in range(self.board_size):
                    if board[x][y] == 0:
                        board[x][y] = 3 - player
                        score = self.minimax(board, depth - 1, True, player, alpha, beta)
                        board[x][y] = 0
                        best_score = min(best_score, score)
                        beta = min(beta, best_score)
                        if beta <= alpha:
                            break
            return best_score

    def is_game_over(self, board):
        return logic.is_game_over(self.player, board) is not None

    def custom_heuristic(self, board, player):
        score = 0
        for row in board:
            score += self.count_pieces_in_line(row, player)
        for col in np.transpose(board):
            score += self.count_pieces_in_line(col, player)
        diagonals = [np.diagonal(board), np.diagonal(np.flipud(board))]
        for diag in diagonals:
            score += self.count_pieces_in_line(diag, player)
        return score

    def count_pieces_in_line(self, line, player):
        count = 0
        for piece in line:
            if piece == player:
                count += 1
            elif piece == 3 - player:
                return 0
        return count

    def evaluate_board(self, board, player):
        if logic.is_game_over(player, board):
            return 10
        elif logic.is_game_over(3 - player, board):
            return -10
        else:
            return self.custom_heuristic(board, player)

    def start(self):
        return self.select_tile(self.root_state, self.player)
class MonteCarloPlayer(PlayerStrat):
    def __init__(self, _board_state, player, time_limit=5):
        super().__init__(_board_state, player)
        self.time_limit = time_limit  # Temps limite pour l'exécution de MCTS

    def start(self):
        root = self.create_node(self.root_state, None, None)
        end_time = time.time() + self.time_limit

        while time.time() < end_time:
            node = root
            state = copy.deepcopy(self.root_state)

            # Phase de sélection et d'expansion
            while node["untried_moves"] == [] and node["children"]:
                node = self.select_child(node)
                state[node["move"]] = self.player

            if node["untried_moves"]:
                move = random.choice(node["untried_moves"])
                state[move] = self.player
                node = self.add_child(node, move, state)

            # Phase de simulation
            result = self.simulate(copy.deepcopy(state))

            # Phase de rétropropagation
            self.backpropagate(node, result)

        return self.get_best_move(root)

    def create_node(self, state, parent, move):
        untried_moves = logic.get_possible_moves(state)  # Liste des mouvements possibles
        return {"state": state, "parent": parent, "move": move, "untried_moves": untried_moves, "children": [], "wins": 0, "visits": 0}

    def select_child(self, node):
        best_score = -float('inf')
        best_child = None
        for child in node["children"]:
            score = child["wins"] / child["visits"] + sqrt(2 * log(node["visits"]) / child["visits"])
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def add_child(self, node, move, state):
        child = self.create_node(state, node, move)
        node["children"].append(child)
        node["untried_moves"].remove(move)
        return child

    def simulate(self, state):
        current_player = self.player
        while not logic.is_game_over(current_player, state):
            possible_moves = logic.get_possible_moves(state)
            if not possible_moves:
                break
            move = random.choice(possible_moves)
            state[move] = current_player
            current_player = logic.BLACK_PLAYER if current_player == logic.WHITE_PLAYER else logic.WHITE_PLAYER
        return 1 if logic.is_game_over(self.player, state) else 0

    def backpropagate(self, node, result):
        while node:
            node["visits"] += 1
            if node["parent"] and node["parent"]["state"][node["move"]] == self.player:
                node["wins"] += result
            node = node["parent"]

    def get_best_move(self, node):
        return max(node["children"], key=lambda child: child["visits"])["move"]
    



# Dictionary to map strategy names to their respective classes.
str2strat: dict[str, PlayerStrat] = {
    "human": None,
    "random": RandomPlayer,
    "minimax": MiniMax,
    "montecarlo": MonteCarloPlayer,
}
