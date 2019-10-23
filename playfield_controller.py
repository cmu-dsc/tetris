from gamestate import GameState
from pieces import *
from playfield import Playfield
from numpy import random
import warnings

class PlayfieldController:
    def __init__(self):
        self._active_piece = None
        self._next_piece_class = None
        self._playfield = Playfield()
        self._score = 0
        self._game_over = False
        self._rng_queue = [] # for _gen_next_piece: we generate a queue of all
                             # 7 pieces, shuffled, and sample as needed. When
                             # the queue is empty, regenerate it.
    def _gen_next_piece(self):
        # based on rng algorithm described in https://tetris.fandom.com/wiki/Random_Generator
        if not self._rng_queue: # if the queue is empty
            self._rng_queue = random.permutation([I, J, L, O, S, T, Z])
        else:
            self._next_piece = self._rng_queue[0]
            self._rng_queue = np.delete(self._rng_queue, 0)

    # These four methods should perform the necessary checking, and transform the active
    # piece according to the official Tetris rules.
    def move_left(self):
        # Has the effect of pressing left on the controller.
        try:
            if self._playfield.is_legal_move(self._active_piece,
                                            (self._active_piece.coords[0] - 1,
                                            self._active_piece.coords[1])):
                self._active_piece.coords =  (self._active_piece.coords[0] - 1,
                                            self._active_piece.coords[1])
        except AttributeError:
            warnings.warn('Tried to move left before starting the game. Updating game.',
                          RuntimeWarning)
            self.update()
            self.move_left()
    def move_right(self):
        '''Has the effect of pressing right on the controller.'''
        try:
            if self._playfield.is_legal_move(self._active_piece,
                                            (self._active_piece.coords[0] + 1,
                                            self._active_piece.coords[1])
                                            ):
                self._active_piece.coords =  (self._active_piece.coords[0] + 1,
                                            self._active_piece.coords[1])
        except AttributeError:
            warnings.warn('Tried to move right before starting the game. Statrting game.',
                          RuntimeWarning)
            self.update()
            self.move_right()
    def rotate_cw(self):
        '''Has the effect of pressing rotate-clockwise on the controller.'''
        # Read about wall kicks here: https://tetris.wiki/SRS#Wall_Kicks
        # O tetrominos don't rotate
        if isinstance(self._active_piece, O):
            pass
        # Try a basic rotation
        try:
            orientation = self._active_piece.orientation
            if self._playfield.is_legal_move(self._active_piece.rotate_cw(),
                                            self._active_piece.coords):
                    pass # nothing to do, we already rotated the piece
            # Try wall kicks: https://tetris.wiki/SRS#Wall_Kicks
            else:
                if isinstance(self._active_piece, I): # I tetrominos have their own wall
                                                      # kick rules.
                    wall_kick_data = [[(-2,  0), ( 1,  0), (-2, -1), ( 1,  2)],
                                      [(-1,  0), ( 2,  0), (-1, +2), ( 2, -1)],
                                      [( 2,  0), (-1,  0), ( 2,  1), (-1, -2)],
                                      [( 1,  0), (-2,  0), ( 1, -2), (-2,  1)]]
                else:
                    wall_kick_data = [[(-1,  0), (-1,  1), ( 0, -2), (-1, -2)],
                                      [( 1,  0), ( 1, -1), ( 0,  2), ( 1,  2)],
                                      [( 1,  0), ( 1,  1), ( 0, -2), ( 1, -2)],
                                      [(-1,  0), (-1, -1), ( 0,  2), (-1,  2)]]
                for test in [0, 1, 2, 3]:
                    if self._playfield.is_legal_move(self._active_piece,
                             (self._active_piece.coords[0] + wall_kick_data[orientation][test][0],
                              self._active_piece.coords[1] + wall_kick_data[orientation][test][1])):
                        break # stop the tests, keep the rotation
                    elif test == 3:
                        # If we've gone through all the tests and
                        # no wall kick yields a legal rotation,
                        # rotate our piece back.
                        self._active_piece.rotate_ccw()
        except AttributeError:
            warnings.warn('Tried to rotate cw before starting the game. Starting game.',
                          RuntimeWarning)
            self.update()
            self.rotate_cw()
    def rotate_ccw(self):
        '''Has the effect of pressing rotate-counterclockwise on the controller.'''
        # Read about wall kicks here: https://tetris.wiki/SRS#Wall_Kicks
        if isinstance(self._active_piece, O):
            pass
        # Try a basic rotation
        try:
            orientation = self._active_piece.orientation
            if self._playfield.is_legal_move(self._active_piece.rotate_ccw(),
                                            self._active_piece.coords):
                    pass # nothing to do, we already rotated the piece
            # Try wall kicks: https://tetris.wiki/SRS#Wall_Kicks
            else:
                if isinstance(self._active_piece, I): # I tetrominos have their own wall
                                                      # kick rules.
                    wall_kick_data = [[( 1,  0), ( 1,  1), ( 0, -2), ( 1, -2)],
                                      [( 1,  0), ( 1, -1), ( 0,  2), ( 1,  2)],
                                      [(-1,  0), (-1,  1), ( 0, -2), (-1, -2)],
                                      [( 1,  0), ( 1,  1), ( 0, -2), ( 1,  1)]]
                else:
                    wall_kick_data = [[(-1,  0), ( 2,  0), (-1,  2), ( 2, -1)],
                                      [( 2,  0), (-1,  0), ( 2,  1), (-1, -2)],
                                      [( 1,  0), (-2,  0), ( 1, -2), (-2,  1)],
                                      [(-2,  0), ( 1,  0), (-2, -1), ( 1,  2)]]
                for test in [0, 1, 2, 3]:
                    if self._playfield.is_legal_move(self._active_piece,
                             (self._active_piece.coords[0] + wall_kick_data[orientation][test][0],
                              self._active_piece.coords[1] + wall_kick_data[orientation][test][1])):
                        break # stop the tests, keep the rotation
                    elif test == 3:
                        # If we've gone through all the tests and
                        # no wall kick yields a legal rotation,
                        # rotate our piece back.
                        self._active_piece.rotate_cw()
        except AttributeError:
            warnings.warn('Tried to rotate ccw before starting the game. Starting game.',
                          RuntimeWarning)
            self.update()
            self.rotate_ccw()

    # This dictionary defines the initial coordinates for each type of piece.
    # There are contradicting definitions for this, so we need to figure out which
    # we will stick with.
    initial_coords = {I : (3, 20), J : (3, 21), L : (3, 21), O : (3, 21),
                      S : (3, 21), T : (3, 21), Z : (3, 21)}
    def update(self):
        '''
        Has the effect of updating the game state. This either means moving the piece down,
        locking in the piece and dropping the new piece/generating a new self._next_piece,
        or ending the game.
        
        Return True or False depending on if the game is over.
        '''
        if self._game_over:
            return True
        else:
            # Proceed...
            # If the action is to end the game, set self._game_over = True and return True.
            #
            # If the _active_piece is None, start the game by genarating the active piece
            # and the next piece class objects, and instantiate the active piece with
            # initial coordinates.
            if self._active_piece == None:
                self._gen_next_piece()
                self._active_piece = self._next_piece(initial_coords[self._next_piece])
                self._gen_next_piece()
            # If we're able to move down, move the active piece one row down.
            if self._playfield.is_legal_move(self._active_piece,
                                                 (self._active_piece.coords[0],
                                                  self._active_piece.coords[1] - 1)):
                self._active_piece.coords((self._active_piece.coords[0],
                                           self._active_piece.coords[1] - 1))
            else:
                # If we can't move down and the row is greater than the top row, then game over
                if self._active_piece.coords[1] > 19:
                    self._game_over = True
                    return True
                # Otherwise:
                else:
                    self._playfield.insert_piece(self._active_piece, self._active_piece.coords)
                    # clear completed rows if necessary
                    points = [40, 100, 300,  1200] # number of points awarded for each
                                                   # successively cleared row
                    # This clears filled rows and drops pieces as necessary
                    num_cleared = self._playfield.clear_filled_rows()
                    assert(num_cleared >= 0)
                    # When pieces are dropped, additional rows may fill up and need cleared.
                    while num_cleared:
                        assert(num_cleared >= 0 and num_cleared < 5)
                        self._score += points[num_cleared - 1]
                        num_cleared = self._playfield.clear_filled_rows()
                    # Drop the next piece
                    self._active_piece = self._next_piece(self._initial_coords[self._next_piece])
                    self._gen_next_piece()
            # The game has not ended. Return false.
            return False
    def gamestate(self):
        return GameState(self._playfield, self._active_piece, self._next_piece)
