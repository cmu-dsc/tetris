def surface_area(board):
    sa = 0
    for row in range(board.shape[0]):
        for col in range(board.shape[1]):
            if board[row, col]:
                for i,j in [(-1,0), (1,0), (0,-1), (0,1)]:
                    if row + i >= 0 and col + j >= 0 and row + i < board.shape[0] and col + j < board.shape[1]:
                        if not board[row + i, col + j]:
                            sa += 1
    return sa