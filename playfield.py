# The class representing the board of inactive pieces.
# This is a numpy array of hexidecimal color strings,
# where colors are as defined in pieces.py. Black ('000000') is
# reserved for empty grid cells.
#
# Coordinate definitions:
#    The board coordinates are defined in matrix-convention, with the origin at the
#    top-left of the board.
#    
#    The local piece coordinates are defined in matrix-convention, with the origin at the
#    top-left of the piece's grid.
# 
import numpy as np
class Playfield:
    def __init__(self):
        self._board = np.array((20, 10), dtype = '<U6')
        self._board.fill('000000')
    @property
    def board(self):
        return self._board
    # This on its own might be useful if we want to see if a rotation is valid.
    def is_legal_move(self, piece, coords):
        '''
        piece: a Piece object

        returns: True if the placement is legal (fits), and False if it is not. The piece is
        not placed if the move is not legal.
        '''
        # Note that coordinates can (and sometimes must) be negative.
        # Hint: a boolean 'and' will check for overlap
        raise NotImplementedError
    def insert_piece(self, piece, coords):
        '''
        Inserts a piece onto the game board if the move is legal.
        
        piece: a Piece object

        returns: True if the placement is legal (fits), and False if it is not. The piece is
        not placed if the move is not legal.
        '''
        # Note that coordinates can (and sometimes must) be negative.
        raise NotImplementedError
    def get_bool_board(self):
        '''
        returns: a numpy boolean array representing the board.
        '''
        return self._board != '000000'
    