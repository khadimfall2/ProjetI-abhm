from typing import List, Optional

import numpy as np
BLACK_PLAYER = 1
WHITE_PLAYER = 2
player2str = {1: 'Black', 2: 'White'}


def get_possible_moves(board: np.ndarray) -> list:
    """
    @return   All the coordinates of nodes where it is possible to
              play.
    """
    (x, y) = np.where(board == 0)
    return list(zip(x, y))


def get_player_tiles(board: np.ndarray, player: int) -> list:
    """
    @return   All the coordinates of nodes where the player played.
    """
    (x, y) = np.where(board == player)
    return list(zip(x, y))


def is_game_over(player: int, board: np.ndarray) -> Optional[int]:
    """
    @return   The winning player:  1 or 2 (or None if the game is
              over by lack of playable position!)
    """
    get_border = (lambda i: (i, 0)) if player == BLACK_PLAYER else (lambda i: (0, i))
    for i in range(board.shape[0]):
        path = traverse(get_border(i), player, board, {})
        if path:
            return player
    return None


def is_border(node: tuple, player: int, board_size: int) -> bool:
    """
    @return   Checks whether the given node is a border that
              belongs to the given player.
    """
    (x, y) = node
    return (
        player == BLACK_PLAYER and y == board_size - 1 or
        player == WHITE_PLAYER and x == board_size - 1
    )


def traverse(node: tuple, player: int, board: np.ndarray,
             visited: dict) -> Optional[list]:
    """
    @return   the path of node connecting two borders for player,
              if existing
    """
    (x, y) = node
    if node in visited:
        return None

    if board[x][y] == player:
        visited[node] = 1

        if is_border(node, player, board.shape[0]):
            return visited

        neighbours = get_neighbours(node, board.shape[0])
        for neighbour in neighbours:
            res = traverse(neighbour, player, board, visited)
            if res:
                return res
    return None


def get_neighbours(coordinates: tuple, board_size: int) -> list:
    """
    @return   a list of the neighbours of "coordinates" node
    """
    (x, y) = coordinates
    return [
        node for row in range(-1, 2) for col in range(-1, 2)
        if row != col
        for node in [ (x + row, y + col) ]
        if is_valid(node, board_size)
    ]


def is_valid(coordinates: tuple, board_size: int) -> bool:
    """
    @return   True iff node exists.
    """
    return all(0 <= c < board_size for c in coordinates)


def is_node_free(coordinates: tuple, board: np.ndarray) -> bool:
    """
    @return   True iff node is free.
    """
    (x, y) = coordinates
    return not board[x][y]
