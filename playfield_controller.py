from playfield import Playfield
from gamestate import GameState
from pieces import *

class PlayfieldController:
    def __init__(self):
        self._active_piece = self._gen_next_piece()
        self._next_piece = self._gen_next_piece()
        self._playfield = Playfield()
        self._score = 0
        self._gamestate = GameState(self._playfield, self._active_piece, self._next_piece)
        self._game_over = False
    def _gen_next_piece(self):
        # The next piece should have coordinates reflecting rules outlined in
        # "Spawn Orientation and Location" from https://tetris.wiki/SRS
        raise NotImplementedError

    # These four methods should perform the necessary checking, and transform the active
    # piece according to the official Tetris rules.
    def move_left(self):
        # Has the effect of pressing left on the controller.
        raise NotImplementedError
    def move_right(self):
        # Has the effect of pressing right on the controller.
        raise NotImplementedError
    def rotate_cw(self):
        # Has the effect of pressing rotate-clockwise on the controller.
        raise NotImplementedError
    def rotate_ccw(self):
        # Has the effect of pressing rotate-counterclockwise on the controller.
        raise NotImplementedError

    def update(self):
        # Has the effect of updating the game state. This either means moving the piece down,
        # locking in the piece and dropping the new piece/generating a new self._next_piece,
        # or ending the game.
        #
        # Return True or False depending on if the game is over.
        if self._game_over:
            return True
        else:
            # Proceed...
            raise NotImplementedError
            # If the action is to end the game, set self._game_over = True and return True.
        