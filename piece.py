import numpy as np

# Abstract base class for tetrominos
class Piece:
    def __init__(self):
        # A hexidecimal color string
        self._color = None
        # A boolean matrix in the piece's own local coordinate system.
        # The I piece needs 4x4, the O needs 3x4, and the rest need 3x3.
        # Defined by: https://tetris.wiki/images/3/3d/SRS-pieces.png
        self._grid = None
    # Rotate left w.r.t. local coords
    def rotate_left(self):
        # "Virtual" method
        if self._grid is not None:
            self._grid = np.rot90(self._grid, k=1)
    # Rotate right w.r.t. local coords
    def rotate_right(self):
        # "Virtual" method
        if self._grid is not None:
            self._grid = np.rot90(self._grid, k=-1)
    @property
    def color(self):
        return self._color
    @property
    def grid(self):
        return self._grid
    def __str__(self):
        s = ""
        for r in self._grid:
            for c in r:
                s += '#' if c else '.'
            s += '\n'
        if len(s) != 0:
            return s[:-1]
        return s
    def __repr__(self):
        return self.__str__()