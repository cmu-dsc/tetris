# This is the object which will serve as the interface to our data analysis code.
class GameState:
    def __init__(self, playfield, active_piece, next_piece):
        self._playfield = playfield
        self._active_piece = active_piece
        self._next_piece = next_piece
    def plot(self):
        raise NotImplementedError