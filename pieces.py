from piece import Piece

class I(Piece):
    def __init__(self):
        self._color = '5FFBFD'
        # Initial position as defined in: https://tetris.wiki/SRS
        self._grid = [[False, False, False, False],
                      [True,  True,  True,  True],
                      [False, False, False, False],
                      [False, False, False, False]]
    def rotate_left():
        # Implement me
    def rotate_right():
        # Implement me

class J(Piece):
    def __init__(self):
        self._color = '0029F5'
        # Initial position as defined in: https://tetris.wiki/SRS
        self._grid = [[True,  False, False],
                      [True,  True,  True],
                      [False, False, False]]
    def rotate_left():
        # Implement me
    def rotate_right():
        # Implement me

class L(Piece):
    def __init__(self):
        self._color = 'F8AC3A'
        # Initial position as defined in: https://tetris.wiki/SRS
        self._grid = [[False, False, True],
                      [True,  True,  True],
                      [False, False, False]]
    def rotate_left():
        # Implement me
    def rotate_right():
        # Implement me

class O(Piece):
    def __init__(self):
        self._color = 'FFFC52'
        # Initial position as defined in: https://tetris.wiki/SRS
        self._grid = [[False, True,  True,  False],
                      [False, True,  True,  False]
                      [False, False, False, False]]
    def rotate_left():
        # Does nothing to the piece
        pass
        # or return self, return grid, etc. depending on how we code the rest
    def rotate_right():
        # Does nothing to the piece
        pass
        # or return self, return grid, etc. depending on how we code the rest

class S(Piece):
    def __init__(self):
        self._color = '65F84B'
        # Initial position as defined in: https://tetris.wiki/SRS
        self._grid = [[False, True,  True],
                      [True,  True,  False],
                      [False, False, False]]
    def rotate_left():
        # Implement me
    def rotate_right():
        # Implement me

class T(Piece):
    def __init__(self):
        self._color = '8D34F6'
        # Initial position as defined in: https://tetris.wiki/SRS
        self._grid = [[False, True,  False],
                      [True,  True,  True],
                      [False, False, False]]
    def rotate_left():
        # Implement me
    def rotate_right():
        # Implement me

class Z(Piece):
    def __init__(self):
        self._color = 'F2361F'
        # Initial position as defined in: https://tetris.wiki/SRS
        self._grid = [[True,  True,  False],
                      [False, True,  True],
                      [False, False, False]]
    def rotate_left():
        # Implement me
    def rotate_right():
        # Implement me