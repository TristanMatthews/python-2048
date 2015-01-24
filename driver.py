#!/usr/bin/python
# -*- coding: utf-8 -*-
#
#   GameDriver.py
#
#   Simple game driver to auto play a game. Currently only written to support 2048,
#   but should work other turn based games with a well defined legal move set.
#
#   Example:
#       metrics_list = ['bias']
#       metrics = game.Metrics2048(metrics_list)
#       sim = driver.LearnGame(game.Game2048, metrics, eta=0.1, start_parameters=[0])
#       sim.train(10000)
#
################################################################

import numpy as np
import scipy as sp


def set_trace():
    from IPython.core.debugger import Pdb
    import sys
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)


def debug(f, *args, **kwargs):
    from IPython.core.debugger import Pdb
    pdb = Pdb(color_scheme='Linux')
    return pdb.runcall(f, *args, **kwargs)


class GameDriver(object):
    """Driver class that automatically plays a game."""

    def __init__(self, game):
        # Initialize Board.
        self.game = game

    def move_print(self, direction):
        """Wrapper function for move_board that prints board and score."""
        self.game.move_board(direction)
        print(self.game.score)
        print(self.game.board)

    def random_move(self):
        """Generate a random move."""
        direction = np.random.choice(self.game.legal_moves)
        self.game.move_board(direction)

    def random_play(self):
        """Play randomly until loss."""
        while not self.game.lost:
            self.game.random_move()
        print(self.game.score)
        print(self.game.board)


class LearnGame(object):
    """Drive the learning algorithm."""

    def __init__(self, game_class, metrics, n_games=1000, eta=0.01, start_parameters=[]):
        """Initialize class."""
        self.game_class = game_class
        self.metrics = metrics
        self.n_games = n_games
        self.eta = eta
        # Bias parameter plus one for each metric.
        if start_parameters:
            parameters = np.array(start_parameters)
        else:
            parameters = np.zeros(len(metrics.metrics)) + 0.01
        self.parameters = parameters
        self.scores = []
        self.parameter_history = []

    def v_board(self, board):
        """Define the operation form of V hat."""
        values = self.metrics.cal_metrics(board)
        return (self.parameters * values).sum()

    def next_move(self, game):
        """Find the next move."""

        max_v = - np.inf
        move = None
        for m in game.legal_moves:
            success, board = game.move_board(m, move=False)
            if success:
                v = self.v_board(board)
                if v > max_v:
                    move = m
                    max_v = v
        return move

    def play_game(self):
        """Play the game through."""
        board_metrics = []
        game = self.game_class()
        while not game.lost:
            # Calculate the metrics for later use.
            board_metrics.append(self.metrics.cal_metrics(game.board))
            m = self.next_move(game)
            if m is None:
                game.lost = True
                break
            game.move_board(m)
        return game.score, board_metrics

    def new_parameters(self, score, board_metrics):
        """Update the parameters function."""

        # This is probably fubar.
        v_train = np.dot(np.array(board_metrics[1:]), self.parameters)
        # Append score as final v_train value.
        v_train = np.append(v_train, score)
        v_hat = np.dot(np.array(board_metrics), self.parameters)

        deltas = self.eta * (np.array(board_metrics).transpose() * (v_train - v_hat)).mean(1)

        return self.parameters + deltas

    def train(self, n_games=False):
        """Train the parameters."""
        if not n_games:
            n_games = self.n_games
        n = 0
        while n < n_games:
            score, board_metrics = self.play_game()
            self.parameters = self.new_parameters(score, board_metrics)
            self.scores.append(score)
            self.parameter_history .append(self.parameters)
            n += 1
            if not (n % 100):
                print n, sp.median(self.scores[-100:]), self.parameters
