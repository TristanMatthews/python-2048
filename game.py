#!/usr/bin/python
# -*- coding: utf-8 -*-
#
#   game.py
#
#   Python implementation of the game 2048.
#
#   This implementation is to be used to train a game playing algorithm,
#   so few UI elements will be implemented.
#
################################################################


import argparse
import sys
import time

import numpy as np
import scipy as sp
from scipy import stats


def shift(vector, n_shift=1, inplace=False):
    """Shift an array by n_shift and fill with zeros."""
    if not inplace:
        vector = vector.copy()
    # Shift elements over to empty space.
    vector[0:-n_shift] = vector[n_shift:]
    # Set last element to 0.
    vector[-n_shift:] = 0
    return vector


class Game2048(object):
    """Base Class for game 2048."""

    ODDS_4 = 0.1

    def __init__(self):
        # Initialize Board.
        self.board = np.zeros([4, 4], dtype=np.int)
        self.score = 0
        self.lost = False
        self.moves = []
        # For 2048 the level moves never change, we just record failed ones.
        self.legal_moves = [0, 1, 2, 3]
        self.failed_moves = set()

        # Initialize board.
        self.initalize_board()

    def add_tile(self, new_board=False):
        """Add a new tile to the board.

        I haven't checked this algorithm against gabrielecirulli.github.io/2048/
        implantation yet. Assume new tile is added to any empty space will equal odds.
        """

        empty = np.where(self.board == 0)
        n_empty = empty[0].shape[0]
        if n_empty == 0:
            self.lost = True
            print('You have lost.')
        i = np.random.randint(0, n_empty)
        if np.random.rand() > self.ODDS_4:
            value = 2
        else:
            value = 4

        if new_board:
            board = self.board.copy()
            board[empty[0][i], empty[1][i]] = value
            return board
        else:
            self.board[empty[0][i], empty[1][i]] = value

    def initalize_board(self):
        """The is initialized by adding 2 tiles."""
        self.add_tile()
        self.add_tile()

    def update_score(self, n):
        """Score a move."""
        self.score += n

    def move_row(self, row, move=True):
        """A row or column is combined by sliding one way.

        All rows and columns will be moved in and slide left.
        """
        row = row.copy()
        i = 0
        while i < 3:
            if row[i] == 0:
                if row[i:].sum():
                    row[i:] = shift(row[i:])
                    continue
                else:
                    break
            else:
                while row[i+1] == 0:
                    if row[i+1:].sum():
                        row[i+1:] = shift(row[i+1:])
                    else:
                        break
                if row[i] == row[i+1]:
                    row[i] = 2 * row[i]
                    row[i+1:] = shift(row[i+1:])
                    if move:
                        self.update_score(row[i])
                i = i+1
        return row

    def move_board(self, direction, move=True):
        """Move all rows / columns in the indicated direction.

        0: left
        1: right
        2: up
        3: down
        """

        # Get working board.
        w_board = self.board.copy()

        # If it is a known bad move simply return.
        if direction in self.failed_moves:
            return
        if direction not in self.legal_moves:
            raise Exception('Invalid move. Try Left (0), Right (1), Up (2), or Down (3).')

        # Cycle through row/columns
        if direction in [0, 2]:
            sign = 1
        else:
            sign = -1

        for i in np.arange(4):
            if direction in [0, 1]:
                w_board[i, :][::sign] = self.move_row(w_board[i, :][::sign], move=move)
            else:
                w_board[:, i][::sign] = self.move_row(w_board[:, i][::sign], move=move)

        success = not np.array_equal(self.board, w_board)

        if move:
            # Update the board if something has changed.
            if success:
                # Record the move.
                self.moves.append(direction)
                self.board = w_board
                self.add_tile()
                # If the move was successful reset failed moves.
                self.failed_moves = set()
            else:
                # Else update failed moves, and check if you have lost.
                self.failed_moves.add(direction)
                if len(self.failed_moves) == 4:
                    self.lost = True
        else:
            return success, w_board


class Metrics2048(object):
    """Calculate the move metrics for 2048."""

    def __init__(self, metrics_list):
        """Initialize the class and game."""
        for metric in metrics_list:
            if not hasattr(self, metric):
                raise Exception("Unknown metric in metrics list %s" % metric)
        self.metrics = metrics_list

    def cal_metrics(self, board):
        """Calculate all requested metrics."""
        values = []
        for metric in self.metrics:
            values.append(getattr(self, metric)(board))
        return np.array(values)

    def bias(self, board):
        """Bias metric."""
        return 1.0

    def n_empty(self, board):
        """Count number of empty cells."""
        return np.where(board == 0)[0].shape[0]

    def n_full(self, board):
        """Count number of filled cells."""
        return np.where(board != 0)[0].shape[0]

    def var(self, board):
        """Calculates boards variance."""
        return np.var(board)

    def std(self, board):
        """Calculates boards std."""
        return np.std(board)



























