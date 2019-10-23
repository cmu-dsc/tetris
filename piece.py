import numpy as np

# Abstract base class for tetrominos
class Piece(object):
    def __init__(self, coords):
        # A boolean matrix in the piece's own local coordinate system.
        # The I piece needs 4x4, the O needs 3x4, and the rest need 3x3.
        # Defined by: https://tetris.wiki/images/3/3d/SRS-pieces.png
        self._grid = None
        # 2-tuple, i.e. (x coordinate, y coordinate)
        self._coords = coords
        # We need to keep track of the rotation orientation. We use a
        # naming convention similar to https://tetris.wiki/SRS#Wall_Kicks
        # 0: spawn orientation, 1: rotated CW from spawn, 2: rotated twice either dir.,
        # 3: rotated CCW from spawn
        self._orientation = 0
    # A hexidecimal color string. Color should be the same for each instance,
    # therefore we make it a static variable.
    _color = None
    @staticmethod
    def color(cls):
        if cls._color is None:
            raise NotImplementedError('Tried to call method on instance of abstract base class Piece.')
        elif cls._color is '000000':
            raise RuntimeError('No class derived from Piece can have the color 0x000000.')
        return cls._color
    # Rotate counter-clockwise w.r.t. local coords
    def rotate_ccw(self):
        # "Virtual" method
        if self._grid is not None:
            self._grid = np.rot90(self._grid, k=1)
            self._orientation = (self._orientation + 1) % 4
    # Rotate clockwise w.r.t. local coords
    def rotate_cw(self):
        # "Virtual" method
        if self._grid is not None:
            self._grid = np.rot90(self._grid, k=-1)
            self._orientation = (self._orientation - 1) % 4
    @property
    def coords(self):
        return self._coords
    @coords.setter
    def coords(self, coords):
        self._coords = coords
    @property
    def grid(self):
        return self._grid
    @property
    def orientation(self):
        return self._orientation
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