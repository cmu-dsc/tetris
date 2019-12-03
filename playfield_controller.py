from gamestate import GameState
from pieces import *
from playfield import Playfield
import numpy as np
import warnings

class PlayfieldController:
    def __init__(self):
        self._active_piece = None
        self._next_piece_class = None
        self._playfield = Playfield()
        self._score = 0
        self._game_over = False
        self._rng_queue = np.array([]) # for _gen_next_piece_class: we generate a queue of all
                                       # 7 pieces, shuffled, and sample as needed. When
                                       # the queue is empty, regenerate it.
    def _gen_next_piece_class(self):
        # based on rng algorithm described in https://tetris.fandom.com/wiki/Random_Generator
        if self._rng_queue.size == 0: # if the queue is empty
            self._rng_queue = np.random.permutation([O])
        self._next_piece_class = self._rng_queue[0]
        self._rng_queue = np.delete(self._rng_queue, 0)

    # These four methods should perform the necessary checking, and transform the active
    # piece according to the official Tetris rules.
    def move_left(self):
        ''' Has the effect of pressing left on the controller.'''
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
            return
        # Try a basic rotation
        try:
            orientation = self._active_piece.orientation
        except AttributeError:
            warnings.warn('Tried to rotate cw before starting the game. Starting game.',
                          RuntimeWarning)
            self.update()
            self.rotate_cw()
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
                    # stop the tests, keep the rotation
                    self._active_piece.coords =\
                        (self._active_piece.coords[0] + wall_kick_data[orientation][test][0],
                            self._active_piece.coords[1] + wall_kick_data[orientation][test][1])
                    break
                else:
                    if test == 3:
                        # If we've gone through all the tests and
                        # no wall kick yields a legal rotation,
                        # rotate our piece back.
                        self._active_piece.rotate_ccw()

    def rotate_ccw(self):
        '''Has the effect of pressing rotate-counterclockwise on the controller.'''
        # Read about wall kicks here: https://tetris.wiki/SRS#Wall_Kicks
        if isinstance(self._active_piece, O):
            return
        # Try a basic rotation
        try:
            orientation = self._active_piece.orientation
        except AttributeError:
            warnings.warn('Tried to rotate ccw before starting the game. Starting game.',
                          RuntimeWarning)
            self.update()
            self.rotate_ccw()
        if self._playfield.is_legal_move(self._active_piece.rotate_ccw(),
                                        self._active_piece.coords):
                pass # nothing to do, we already rotated the piece
        # Try wall kicks: https://tetris.wiki/SRS#Wall_Kicks
        else:
            if isinstance(self._active_piece, I): # I tetrominos have their own wall
                                                    # kick rules.
                wall_kick_data = [[(-1,  0), ( 2,  0), (-1,  2), ( 2, -1)],
                                    [( 2,  0), (-1,  0), ( 2,  1), (-1, -2)],
                                    [( 1,  0), (-2,  0), ( 1, -2), (-2,  1)],
                                    [(-2,  0), ( 1,  0), (-2, -1), ( 1,  2)]]
            else:
                wall_kick_data = [[( 1,  0), ( 1,  1), ( 0, -2), ( 1, -2)],
                                    [( 1,  0), ( 1, -1), ( 0,  2), ( 1,  2)],
                                    [(-1,  0), (-1,  1), ( 0, -2), (-1, -2)],
                                    [(-1,  0), (-1, -1), ( 0,  2), (-1,  2)]]
            for test in [0, 1, 2, 3]:
                if self._playfield.is_legal_move(self._active_piece,
                            (self._active_piece.coords[0] + wall_kick_data[orientation][test][0],
                            self._active_piece.coords[1] + wall_kick_data[orientation][test][1])):
                    # stop the tests, keep the rotation
                    self._active_piece.coords =\
                        (self._active_piece.coords[0] + wall_kick_data[orientation][test][0],
                            self._active_piece.coords[1] + wall_kick_data[orientation][test][1])
                    break
                else:
                    if test == 3:
                        # If we've gone through all the tests and
                        # no wall kick yields a legal rotation,
                        # rotate our piece back.
                        self._active_piece.rotate_cw()

    # This dictionary defines the initial coordinates for each type of piece.
    # There are contradicting definitions for this, so we need to figure out which
    # we will stick with.
    initial_coords = {I : (3, 17), J : (3, 17), L : (3, 17), O : (3, 17),
                      S : (3, 17), T : (3, 17), Z : (3, 17)}
    def update(self):
        '''
        Has the effect of updating the game state. This either means moving the piece down,
        locking in the piece and dropping the new piece/generating a new self._next_piece_class,
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
                self._gen_next_piece_class()
                self._active_piece = self._next_piece_class(self.initial_coords[self._next_piece_class])
                self._gen_next_piece_class()
            # If we're able to move down, move the active piece one row down.
            if self._playfield.is_legal_move(self._active_piece,
                                                 (self._active_piece.coords[0],
                                                  self._active_piece.coords[1] - 1)):
                self._active_piece.coords = (self._active_piece.coords[0],
                                             self._active_piece.coords[1] - 1)
            # Otherwise, place the piece, etc.
            else:
                self._playfield.insert_piece(self._active_piece, self._active_piece.coords)
                # clear completed rows if necessary
                points = [0, 40, 100, 300,  1200] # number of points awarded for each
                                                  # successively cleared row
                # This clears filled rows and drops pieces as necessary. Returns
                # number of rows cleared.
                num_cleared = self._playfield.clear_filled_rows()
                assert(num_cleared >= 0 and num_cleared < 5)
                self._score += points[num_cleared]
                # Drop the next piece
                self._active_piece = self._next_piece_class(self.initial_coords[self._next_piece_class])
                self._gen_next_piece_class()
                # If we can't place the piece, game over
                if not self._playfield.is_legal_move(self._active_piece, self._active_piece.coords):
                    self._game_over = True
                    return True
            # The game has not ended. Return false.
            return False
    def gamestate(self):
        return GameState(self._playfield, self._active_piece, self._next_piece_class)
    @property
    def game_over(self):
        return self._game_over
