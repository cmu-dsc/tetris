# The class representing the board of inactive pieces.
# This is a numpy array of hexidecimal color strings,
# where colors are as defined in pieces.py. Black ('000000') is
# reserved for empty grid cells.
#
# Coordinate definitions:
#    Cartesian coordinates, with the origin at the lower-left.
#    The first array index is the horizontal coordinate (x),
#    and the second array index is the vertical coordinate (y).
#    This is in contrast to matrix convention.
import numpy as np
class Playfield:
    def __init__(self):
        self._board = np.full((10, 20), '000000', dtype = '<U6')
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
        
        # First check to see if any part of the piece is outside of the board
        # quick optimization -- no need to check if the piece grid is entirely in the board
        #
        # We may need to change the grid, but we do not want to mutate piece:
        grid = piece.grid
        temp_board = np.full((10, 20), False)
        for x in range(0, grid.shape[0]):
            for y in range(0, grid.shape[1]):
                x_lab = x + coords[0]
                y_lab = y + coords[1]
                if (x_lab < 0 or x_lab > 9 or y_lab < 0 or y_lab > 20):
                    if piece.grid[x, y]:
                        return False
                elif y_lab != 20:
                    temp_board[x_lab, y_lab] = piece.grid[x, y]
        return not np.any(np.logical_and(self.get_bool_board(), temp_board))
    def insert_piece(self, piece, coords):
        '''
        Inserts a piece onto the game board if the move is legal.
        
        piece: a Piece object

        returns: True if the placement is legal (fits), and False if it is not. The piece is
        not placed if the move is not legal.
        '''
        # Note that coordinates can (and sometimes must) be negative.
        if self.is_legal_move(piece, coords):
            for x in range(0, piece.grid.shape[0]):
                for y in range(0, piece.grid.shape[1]):
                    x_lab = x + coords[0]
                    y_lab = y + coords[1]
                    if piece.grid[x, y]:
                        self._board[x_lab, y_lab] = piece.color()
            return True
        return False

    def clear_filled_rows(self):
        '''
        Updates board with rows cleared and blocks shifted downward
        accordingly.
        '''
        bool_board = self.get_bool_board()
        y_full_rows = [] # y coordinates of filled rows
        for y in range(bool_board.shape[1]):
            if np.all(bool_board[:, y]):
                y_full_rows.append(y)
        for i in range(len(y_full_rows)):
            self._board[:, y_full_rows[i]:-1] = self._board[:, y_full_rows[i] + 1:]
            self._board[:, -1] = 10 * ['000000']
            for j in range(i + 1, len(y_full_rows)):
                y_full_rows[j] -= 1
        return len(y_full_rows)

    def get_bool_board(self):
        '''
        returns: a numpy boolean array representing the board.
        '''
        return self._board != '000000'
    