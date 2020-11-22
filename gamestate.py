# This is the object which will serve as the interface to our data analysis code.
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
# credit https://gist.github.com/matthewkremer/3295567
def hex_to_rgb(hex):
     hex = hex.lstrip('#')
     hlen = len(hex)
     return tuple(int(hex[i:i+hlen//3], 16) for i in range(0, hlen, hlen//3))
class GameState:
    def __init__(self, playfield, active_piece, next_piece):
        self._playfield = deepcopy(playfield)
        self._active_piece = deepcopy(active_piece)
        self._next_piece = deepcopy(next_piece)
    def plot(self, filename=None):
        # filename is the name of the image to save
        # pad the board
        board = np.full((18, 28), '000000', dtype = '<U6')
        board[4:14, 4:24] = self._playfield.board
        active_piece = np.full(self._active_piece.grid.shape, '000000', dtype = '<U6')
        shape = active_piece.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                if self._active_piece.grid[i, j]:
                    active_piece[i, j] = self._active_piece.color()
        b =  board[self._active_piece.coords[0] + 4:
                       self._active_piece.coords[0] + 4 + self._active_piece.grid.shape[0],
                   self._active_piece.coords[1] + 4 :
                       self._active_piece.coords[1] + 4 + self._active_piece.grid.shape[1]]
        board[self._active_piece.coords[0] + 4:
                  self._active_piece.coords[0] + 4 + self._active_piece.grid.shape[0],
              self._active_piece.coords[1] + 4 :
                  self._active_piece.coords[1] + 4 + self._active_piece.grid.shape[1]] = \
            np.where(active_piece != '000000', active_piece, b)
        im = np.array([[hex_to_rgb(block) for block in row] for row in board])
        im = np.rot90(im, k = 1)
        im = im[4:-4, 4:-4]
        plt.imshow(im)
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
