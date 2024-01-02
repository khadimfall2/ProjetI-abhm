import sys
from typing import Tuple, Optional
import time
import random
import copy
from math import sqrt, log
import pygame
import numpy as np
from rich.table import Table
from rich.console import Console

from classes.ui import UI, NoUI
import classes.logic as logic
from classes.strategy import str2strat


class Game:
    def __init__(self, board_size: int, strat: str,
                 black_starts: bool = True, use_ui=True):
        """
        Initialisation of a new game with:
            * the size of the board,
            * the players strategies, eg., ("human", "random"),
            * which player starts, i.e., black (by default) or white.

        Besides, the user interface is initialised and displayed.

        Also, public variables are set to their initial values:
            * the board is currently empty, an zero-filled 2D array.
            * there is no current winner (set to None), which is to
              become eventually either 1 or 2, respectively for the
              black and white player.

        Finally, a dictionary allows to retrieve the player based on
        the parity (even/odd) of the current step in the game.
        """

        self.strat = strat

        # Initialize info about player and turns
        # Does BLACK player start?
        self.black_starts = black_starts
        self.turn_state = black_starts

        self.turn = { True:  logic.BLACK_PLAYER,
                      False: logic.WHITE_PLAYER }
        black_strat, white_strat = self.strat
        self.strategies = {
            logic.BLACK_PLAYER: black_strat,
            logic.WHITE_PLAYER: white_strat
        }

        # Instantiate classes
        self.use_ui = use_ui
        if self.use_ui:
            self.ui = UI(board_size)
        else:
            self.ui = NoUI(board_size)

        # Initialize public variables
        self.winner = None
        self.board_size = board_size

        # State of the board
        self.logger = np.zeros(
            shape=(self.board_size, self.board_size),
            dtype=np.int8
        )

    def print_game_info(self, args) -> None:
        """
        Prints on the console the parameters of the game:
           * the board size,
           * the players strategies, eg., ("human", "random"),
           * the number of the game when in competition mode.
        """
        if not self.use_ui:
            return

        console = Console()

        table = Table(
            title="Polyline",
            show_header=True,
            header_style="bold cyan"
        )
        table.add_column("Parameters", justify="center")
        table.add_column("Value", justify="right")
        table.add_row("Board size", str(args[0]))
        table.add_row("Mode", str('_'.join(args[1])))
        table.add_row("Game", str(args[2]))

        console.print(table)

    def play(self) -> None:
        """
        Draw the board and deal with user inputs.
        Then play a turn.
        """

        self.ui.draw(
            self.strat,
            self.get_current_strategy()
        )
        self.ui.handle_events(self.strat)
        self.run_turn()

    def run_turn(self) -> bool:
        """
        Run the current player strategie, place the tile on the board,
        check for a winner and initialize the next turn.
        Forbids playing on an already busy node by failing.

        @bug   Notice that this can lead to infinite loops if the
               player always plays an invalid node!
        """
        player = self.turn[self.turn_state]
        strategy_name = self.strategies[player]

        if strategy_name == "human":
            # human player's turn
            node = self.human_turn()
            if node is None:
                # Player did not click yet, no turn to play
                return
            assert logic.is_node_free(node, self.logger), "Human returned a busy node"
        else:
            # AI player's turn
            node = self.ai_turn(player, strategy_name)
            assert logic.is_node_free(node, self.logger), "AI returned a busy node"

        # Place the tile on the node
        self.ui.update_tile_color(node, player)
        x, y = node
        self.logger[x][y] = player

        # Next turn
        self.turn_state = not self.turn_state
        self.ui.last_clicked_node = None

        self.winner = logic.is_game_over(player, self.logger)

    def human_turn(self):
        """Validates a human tile selection.
        Empties the ui.last_clicked_node variable if node is
        unavailable.
        
        Returns:
            int: index of a valid and unoccupied node on the board
        """
        node = self.ui.last_clicked_node
        if node is None:
            return None

        if not logic.is_valid(node, self.logger.shape[0]):
            self.ui.last_clicked_node = None
            return None

        if not logic.is_node_free(node, self.logger):
            self.ui.last_clicked_node = None
            return None

        return node

    def ai_turn(self, player, strategy_name):
        """Runs AI strategie.
        
        Args:
            player: either logic.BLACK_PLAYER or logic.WHITE_PLAYER
            strategy_name (str): name of the strategy
        
        Returns:
            int: index of a valid and unoccupied node on the board
        """
        # Retrieve the proper strategy constructor (cf. strategy.py)
        StrategyConstructor = str2strat[strategy_name]
        strategy = StrategyConstructor(
            _board_state=self.logger,
            player=player
        )
        return strategy.start()

    def get_current_player(self):
        return self.turn[self.turn_state]

    def get_current_strategy(self):
        return self.strategies[self.get_current_player()]

