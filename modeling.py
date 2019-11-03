from playfield import Playfield
from torch import nn
import torch
import pieces

def get_inactive_board(gamestate):
    return gamestate._playfield.get_bool_board().astype(float)

def get_active_board(gamestate):
    playfield = Playfield()
    print(gamestate._active_piece.coords)
    playfield.insert_piece(gamestate._active_piece, gamestate._active_piece.coords)
    return playfield.get_bool_board().astype(float)

def get_board_state(gamestate):
    return torch.stack((torch.tensor(get_inactive_board(gamestate)), torch.tensor(get_active_board(gamestate)))).float()

def next_piece(gamestate):
    piece = gamestate._next_piece
    if isinstance(piece, pieces.I):
        return 0
    if isinstance(piece, pieces.J):
        return 1
    if isinstance(piece, pieces.L):
        return 2
    if isinstance(piece, pieces.O):
        return 3
    if isinstance(piece, pieces.S):
        return 4
    if isinstance(piece, pieces.T):
        return 5
    if isinstance(piece, pieces.Z):
        return 6

def get_piece_state(gamestate):
    result = torch.zeros(7).float()
    result[next_piece(gamestate)] = 1.0
    return result


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(2,4,3,2,1)
        self.conv2 = nn.Conv2d(4,8,3,2,1)
        self.fc1 = nn.Linear(127,100)
        self.fc2 = nn.Linear(100,4)
    def forward(self, board, piece):
        board = self.conv1(board)
        board = self.relu(board)
        board = self.conv2(board)
        board = self.relu(board)
        board = board.reshape(board.shape[0],-1)
        out = torch.cat([board,piece],1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out
        
        
