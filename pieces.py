import numpy as np
from piece import Piece

class I(Piece):
    _color = '5FFBFD'
    def __init__(self):
        # Initial position as defined in: https://tetris.wiki/SRS
        self._grid = np.array(
            [[False, False, False, False],
             [True,  True,  True,  True],
             [False, False, False, False],
             [False, False, False, False]]
        )

class J(Piece):
    _color = '0029F5'
    def __init__(self):
        # Initial position as defined in: https://tetris.wiki/SRS
        self._grid = np.array(
            [[True,  False, False],
             [True,  True,  True],
             [False, False, False]]
        )

class L(Piece):
    _color = 'F8AC3A'
    def __init__(self):
        # Initial position as defined in: https://tetris.wiki/SRS
        self._grid = np.array(
            [[False, False, True],
             [True,  True,  True],
             [False, False, False]]
        )

class O(Piece):
    _color = 'FFFC52'
    def __init__(self):
        # Initial position as defined in: https://tetris.wiki/SRS
        self._grid = np.array(
            [[False, True,  True,  False],
             [False, True,  True,  False],
             [False, False, False, False]]
        )
    def rotate_ccw(self):
        # Does nothing to the piece
        pass
        # or return self, return grid, etc. depending on how we code the rest
    def rotate_cw(self):
        # Does nothing to the piece
        pass
        # or return self, return grid, etc. depending on how we code the rest

class S(Piece):
    _color = '65F84B'
    def __init__(self):
        # Initial position as defined in: https://tetris.wiki/SRS
        self._grid = np.array(
            [[False, True,  True],
             [True,  True,  False],
             [False, False, False]]
        )

class T(Piece):
    _color = '8D34F6'
    def __init__(self):
        # Initial position as defined in: https://tetris.wiki/SRS
        self._grid = np.array(
            [[False, True,  False],
             [True,  True,  True],
             [False, False, False]]
        )

class Z(Piece):
    _color = 'F2361F'
    def __init__(self):
        # Initial position as defined in: https://tetris.wiki/SRS
        self._grid = np.array(
            [[True,  True,  False],
             [False, True,  True],
             [False, False, False]]
        )