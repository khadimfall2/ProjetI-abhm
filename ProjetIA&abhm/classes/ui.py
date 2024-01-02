import sys
from typing import Optional
from math import cos, sin, pi, radians

import pygame
import numpy as np
from pygame import time
from pygame import gfxdraw

import classes.logic as logic


class NoUI:
    def __init__(self, board_size: int):
        self.last_clicked_node = None
    def draw(self, strat, current_strategie):
        pass
    def update_tile_color(self, coordinates: tuple, player: Optional[int]):
        pass
    def handle_events(self, strat) -> None:
        pass


class UI:
    def __init__(self, board_size: int):
        self.board_size = board_size
        assert 1 < self.board_size <= 26

        self.clock = time.Clock()
        self.hex_radius = 20
        self.x_offset, self.y_offset = 60, 60
        self.text_offset = 45
        self.screen = pygame.display.set_mode(
            (self.x_offset + (2 * self.hex_radius) * self.board_size + self.hex_radius * self.board_size,
             round(self.y_offset + (1.75 * self.hex_radius) * self.board_size))
        )

        # Colors
        self.red = (222, 29, 47)
        self.blue = (0, 121, 251)
        self.green = (0, 255, 0)
        self.white = (255, 255, 255)
        self.black = (40, 40, 40)
        self.gray = (70, 70, 70)
        self.bg = (249,224,167)

        self.screen.fill(self.gray)
        self.fonts = pygame.font.SysFont("Sans", 20)

        self.hex_lookup = {}
        self.rects = []
        self.color = [self.bg] * (self.board_size ** 2)
        self.last_clicked_node = None

        self.player2color = {
            logic.BLACK_PLAYER: self.black,
            logic.WHITE_PLAYER: self.white
        }

    # Drawing functions

    def _draw_hexagon(self, surface: object, color: tuple,
                     position: tuple, node: int):
        # Vertex count and radius
        n = 6
        x, y = position
        offset = 3

        # Outline
        self.hex_lookup[node] = [
            (x + (self.hex_radius + offset) * cos(radians(90) + 2 * pi * _ / n),
             y + (self.hex_radius + offset) * sin(radians(90) + 2 * pi * _ / n))
            for _ in range(n)
        ]
        gfxdraw.aapolygon(
            surface, self.hex_lookup[node], color
        )

        # Shape
        gfxdraw.filled_polygon(
            surface,
            [(x + self.hex_radius * cos(radians(90) + 2 * pi * _ / n),
              y + self.hex_radius * sin(radians(90) + 2 * pi * _ / n))
             for _ in range(n)],
            self.color[node]
        )

        # Antialiased shape outline
        gfxdraw.aapolygon(
            surface,
            [(x + self.hex_radius * cos(radians(90) + 2 * pi * _ / n),
              y + self.hex_radius * sin(radians(90) + 2 * pi * _ / n))
            for _ in range(n)],
            self.gray)

        # Placeholder
        rect = pygame.draw.rect(
            surface,
            self.color[node],
            pygame.Rect(
                x - self.hex_radius + offset,
                y - (self.hex_radius / 2),
                (self.hex_radius * 2) - (2 * offset),
                self.hex_radius
            )
        )
        self.rects.append(rect)

        # Bounding box (colour-coded)
        bbox_offset = [0, 3]

        # Top side
        if 0 < node < self.board_size:
            points = (
                [self.hex_lookup[node - 1][3][_] - bbox_offset[_] for _ in range(2)],
                [self.hex_lookup[node - 1][4][_] - bbox_offset[_] for _ in range(2)],
                [self.hex_lookup[node][3][_] - bbox_offset[_] for _ in range(2)]
            )
            gfxdraw.filled_polygon(surface, points, self.white)
            gfxdraw.aapolygon(surface, points, self.white)

        # Bottom side
        if self.board_size ** 2 - self.board_size < node < self.board_size ** 2:
            points = (
                [self.hex_lookup[node - 1][0][_] + bbox_offset[_] for _ in range(2)],
                [self.hex_lookup[node - 1][5][_] + bbox_offset[_] for _ in range(2)],
                [self.hex_lookup[node][0][_] + bbox_offset[_] for _ in range(2)]
            )
            gfxdraw.filled_polygon(surface, points, self.white)
            gfxdraw.aapolygon(surface, points, self.white)

        # Left side
        bbox_offset = [3, -3]

        if node % self.board_size == 0:
            if node >= self.board_size:
                points = (
                    [self.hex_lookup[node - self.board_size][1][_] - bbox_offset[_] for _ in range(2)],
                    [self.hex_lookup[node - self.board_size][0][_] - bbox_offset[_] for _ in range(2)],
                    [self.hex_lookup[node][1][_] - bbox_offset[_] for _ in range(2)]
                )
                gfxdraw.filled_polygon(surface, points, self.black)
                gfxdraw.aapolygon(surface, points, self.black)

        # Right side
        if (node + 1) % self.board_size == 0:
            if node > self.board_size:
                points = (
                    [self.hex_lookup[node - self.board_size][4][_] + bbox_offset[_] for _ in range(2)],
                    [self.hex_lookup[node - self.board_size][5][_] + bbox_offset[_] for _ in range(2)],
                    [self.hex_lookup[node][4][_] + bbox_offset[_] for _ in range(2)]
                )
                gfxdraw.filled_polygon(surface, points, self.black)
                gfxdraw.aapolygon(surface, points, self.black)

    def _draw_text(self):
        alphabet = list(map(chr, range(97, 123)))

        for _ in range(self.board_size):
            # Columns
            text = self.fonts.render(
                alphabet[_].upper(), True, self.white, self.gray)
            text_rect = text.get_rect()
            text_rect.center = (
                self.x_offset + (2 * self.hex_radius) * _,
                self.text_offset / 2
            )
            self.screen.blit(text, text_rect)

            # Rows
            text = self.fonts.render(
                str(_), True, self.white, self.gray)
            text_rect = text.get_rect()
            text_rect.center = (
                (self.text_offset / 4 + self.hex_radius * _,
                 self.y_offset + (1.75 * self.hex_radius) * _)
            )
            self.screen.blit(text, text_rect)

    def draw(self, strat, current_strategie):
        """Draws the board.
        
        Displays the background and info of the game.
        
        Args:
            strat (int): Playing strategies (is there a human playing ?)
            current_strategie ([type]): Current player strategie
        """
        self._limit_framerate(strat, current_strategie)
        self._draw_board()

    def _draw_board(self):
        counter = 0
        for row in range(self.board_size):
            for column in range(self.board_size):
                self._draw_hexagon(
                    self.screen, self.gray,
                    self._get_coordinates(row, column), counter
                )
                counter += 1
        self._draw_text()

    def _limit_framerate(self, strat, current_strategie):
        if 'human' in strat:
            # In human_vs_ai, limit framerate to 25FPS
            if current_strategie != 'human':
                # Wait a little bit before AI's turn (limiting to 10FPS)
                self.clock.tick(10)
            else:
                self.clock.tick(25)
        else:
            # In ai_vs_ai, slow down to actually see
            self.clock.tick(25)
            pass

    # From UI to logic

    def _get_coordinates(self, row: int, column: int):
        x = self.x_offset + (2 * self.hex_radius) * column + self.hex_radius * row
        y = self.y_offset + (1.75 * self.hex_radius) * row

        return x, y

    def _get_true_coordinates(self, node: int):
        return int(node / self.board_size), node % self.board_size

    def _get_selected_node(self):
        mouse_pos = pygame.mouse.get_pos()
        for _, rect in enumerate(self.rects):
            if rect.collidepoint(mouse_pos):
                return _
        return None

    def _display_mouse_node_hover(self):
        # Source: https://bit.ly/2Wl5Grz

        # Get actual node
        node = self._get_selected_node()
        if node is None:
            return

        # Display hovering informations (coordinates)
        row = int(node / self.board_size)
        column = node % self.board_size
        self._draw_hexagon(
            self.screen, self.black,
            self._get_coordinates(row, column), node
        )

        # Text
        x, y = self._get_true_coordinates(node)
        x, y = self._get_coordinates(x, y)
        alphabet = list(map(chr, range(97, 123)))
        txt = alphabet[column].upper() + str(row)
        node_font = pygame.font.SysFont("Sans", 18)
        foreground = self.black if self.color[node] is self.white else self.white
        text = node_font.render(txt, True, foreground, self.color[node])
        text_rect = text.get_rect()
        text_rect.center = (x, y)
        self.screen.blit(text, text_rect)

        return node

    def handle_events(self, strat: str) -> None:
        """
        Updates UI logic according to mouse events:
        hovering, quitting, clicking.
        """
        EVENT = None
        # We can quit at any time
        for event in pygame.event.get():
            if (event.type == pygame.QUIT
                or (event.type == pygame.KEYDOWN
                    and event.key == pygame.K_ESCAPE)):
                EVENT = "quit"
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONUP:
                # User has clicked
                EVENT = "click"

        if 'human' in strat:
            # Get hovering tile
            selected_node = self._display_mouse_node_hover()
            if selected_node is not None and EVENT == "click":
                # Set clicked tile
                node_coord = self._get_true_coordinates(selected_node)
                self.last_clicked_node = node_coord

        pygame.display.update()


    # Update UI representation

    def update_tile_color(self, coordinates: tuple,
                          player: Optional[int]):
        """
        This procedure updates the ui by applying the given action
        of the player at the given coordinates of the board.
        """
        (x, y) = coordinates
        node = x * self.board_size + y

        if player is None:
            player = logic.BLACK_PLAYER

        try:
            self.color[node] = self.player2color[player]
        except KeyError:
            raise KeyError("player is neither white nor black")
