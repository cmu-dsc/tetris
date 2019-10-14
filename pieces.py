import numpy as np
from piece import Piece

class I(Piece):
    def __init__(self):
        self._color = '5FFBFD'
        # Initial position as defined in: https://tetris.wiki/SRS
        self._grid = np.array(
            [[False, False, False, False],
             [True,  True,  True,  True],
             [False, False, False, False],
             [False, False, False, False]]
        )

class J(Piece):
    def __init__(self):
        self._color = '0029F5'
        # Initial position as defined in: https://tetris.wiki/SRS
        self._grid = np.array(
            [[True,  False, False],
             [True,  True,  True],
             [False, False, False]]
        )

class L(Piece):
    def __init__(self):
        self._color = 'F8AC3A'
        # Initial position as defined in: https://tetris.wiki/SRS
        self._grid = np.array(
            [[False, False, True],
             [True,  True,  True],
             [False, False, False]]
        )

class O(Piece):
    def __init__(self):
        self._color = 'FFFC52'
        # Initial position as defined in: https://tetris.wiki/SRS
        self._grid = np.array(
            [[False, True,  True,  False],
             [False, True,  True,  False],
             [False, False, False, False]]
        )
    def rotate_left(self):
        # Does nothing to the piece
        pass
        # or return self, return grid, etc. depending on how we code the rest
    def rotate_right(self):
        # Does nothing to the piece
        pass
        # or return self, return grid, etc. depending on how we code the rest

class S(Piece):
    def __init__(self):
        self._color = '65F84B'
        # Initial position as defined in: https://tetris.wiki/SRS
        self._grid = np.array(
            [[False, True,  True],
             [True,  True,  False],
             [False, False, False]]
        )

class T(Piece):
    def __init__(self):
        self._color = '8D34F6'
        # Initial position as defined in: https://tetris.wiki/SRS
        self._grid = np.array(
            [[False, True,  False],
             [True,  True,  True],
             [False, False, False]]
        )

class Z(Piece):
    def __init__(self):
        self._color = 'F2361F'
        # Initial position as defined in: https://tetris.wiki/SRS
        self._grid = np.array(
            [[True,  True,  False],
             [False, True,  True],
             [False, False, False]]
        )