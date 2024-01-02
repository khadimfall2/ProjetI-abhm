import logging
import argparse

from typing import Optional

from rich import print
from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET",
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler()]
)

from classes.strategy import str2strat
from classes.tournament import Tournament


def main(args):
    """
    Runs a tournament with a list of arguments that contain, in order:
       * the size of the board,
       * the player strategies , i.e., "human", "random", "minimax", .
       * the game counter (why not? though it should be always zero),
       * the number of games to play.

    If there is only AIs , there is a real competition.
    In contrast, if there's a "human", there is a single match, i.e.,
    the last parameter is ineffective.
    """
    arena = Tournament(args)

    STRAT = args[1]

    if 'human' in STRAT:
        arena.single_game(black_starts=True)
    else:
        arena.championship()


def arguments():
    parser = argparse.ArgumentParser(description='Runs a game of Hex.')

    parser.add_argument(
        '--size', default=7, type=int,
        help='Size of the board (Default: 7)'
    )

    parser.add_argument(
        '--games', default=5, type=int,
        help='Number of games to play in tournament. Only if no human'\
             ' (default: 5)'
    )

    parser.add_argument(
        '--no-ui', action='store_true',
        help='GUI is not displayed. Only if no human.'
    )

    parser.add_argument(
        '--player', default='human', choices=str2strat,
        help='Strategy for player1 (default: human)'
    )
    parser.add_argument(
        '--other', default='random', choices=str2strat,
        help='Strategy for player2 (default: random)'
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    log = logging.getLogger("rich")

    args = arguments()

    STRAT = [args.player, args.other]
    BOARD_SIZE = args.size
    GAME_COUNT = 0
    N_GAMES    = args.games
    USE_UI     = 'human' in STRAT or not args.no_ui

    main([ BOARD_SIZE, STRAT, GAME_COUNT, N_GAMES, USE_UI ])
