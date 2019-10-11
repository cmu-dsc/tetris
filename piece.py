# Abstract base class for tetrominos
class Piece:
    def __init__(self):
        # A hexidecimal color string
        self._color = None
        # A boolean matrix in the piece's own local coordinate system.
        # 3x3 is sufficient for all but the I piece, which needs 4x4.
        # Define by: https://tetris.wiki/images/3/3d/SRS-pieces.png
        self._grid = None
    # Rotate left w.r.t. local coords
    def rotate_left(self):
        # "Virtual" method
        raise NotImplementedError()
    # Rotate right w.r.t. local coords
    def rotate_right(self):
        # "Virtual" method
        raise NotImplementedError()
    @property
    def color(self):
        return self._color
    @color.setter
    def color(self, color):
        self._color = color
    @property
    def grid(self):
        return self._grid