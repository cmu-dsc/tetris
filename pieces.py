import numpy as np
from piece import Piece

class I(Piece):
    _color = '5FFBFD'
    def __init__(self, coords):
        super().__init__(coords)
        # Initial position as defined in: https://tetris.wiki/SRS
        self._grid = np.rot90(np.array(
            [[False, False, False, False],
             [True,  True,  True,  True],
             [False, False, False, False],
             [False, False, False, False]]), k = -1
        )

class J(Piece):
    _color = '0029F5'
    def __init__(self, coords):
        super().__init__(coords)
        # Initial position as defined in: https://tetris.wiki/SRS
        self._grid = np.rot90(np.array(
            [[True,  False, False],
             [True,  True,  True],
             [False, False, False]]), k = -1
        )

class L(Piece):
    _color = 'F8AC3A'
    def __init__(self, coords):
        super().__init__(coords)
        # Initial position as defined in: https://tetris.wiki/SRS
        self._grid = np.rot90(np.array(
            [[False, False, True],
             [True,  True,  True],
             [False, False, False]]), k = -1
        )

class O(Piece):
    _color = 'FFFC52'
    def __init__(self, coords):
        super().__init__(coords)
        # Initial position as defined in: https://tetris.wiki/SRS
        self._grid = np.rot90(np.array(
            [[False, True,  True,  False],
             [False, True,  True,  False],
             [False, False, False, False]]), k = -1
        )
    def rotate_ccw(self):
        # Does nothing to the piece
        return self
        # or return self, return grid, etc. depending on how we code the rest
    def rotate_cw(self):
        # Does nothing to the piece
        return self
        # or return self, return grid, etc. depending on how we code the rest

class S(Piece):
    _color = '65F84B'
    def __init__(self, coords):
        super().__init__(coords)
        # Initial position as defined in: https://tetris.wiki/SRS
        self._grid = np.rot90(np.array(
            [[False, True,  True],
             [True,  True,  False],
             [False, False, False]]), k = -1
        )

class T(Piece):
    _color = '8D34F6'
    def __init__(self, coords):
        super().__init__(coords)
        # Initial position as defined in: https://tetris.wiki/SRS
        self._grid = np.rot90(np.array(
            [[False, True,  False],
             [True,  True,  True],
             [False, False, False]]), k = -1
        )

class Z(Piece):
    _color = 'F2361F'
    def __init__(self, coords):
        super().__init__(coords)
        # Initial position as defined in: https://tetris.wiki/SRS
        self._grid = np.rot90(np.array(
            [[True,  True,  False],
             [False, True,  True],
             [False, False, False]]), k = -1
        )