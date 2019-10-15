from playfield import Playfield
from gamestate import GameState
from pieces import *

class PlayfieldController:
    def __init__(self):
        self._active_piece = None
        self._next_piece = None
        self._playfield = Playfield()
        self._score = 0
        self._game_over = False
    def _gen_next_piece(self):
        # returns class object (not class instance!)
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
        # When dropping the new piece, the next piece should have coordinates reflecting
        # rules outlined in "Spawn Orientation and Location" from https://tetris.wiki/SRS
        #
        # Return True or False depending on if the game is over.
        if self._game_over:
            return True
        else:
            # Proceed...
            raise NotImplementedError
            # If the action is to end the game, set self._game_over = True and return True.
            #
            # If the _active_piece is None, start the game by genarating the active piece
            # and the next piece class objects, and instantiate the active piece with
            # initial coordinates.
    def gamestate(self):
        return GameState(self._playfield, self._active_piece, self._next_piece)