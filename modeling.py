from playfield import Playfield
from torch import nn
import torch
import pieces
import numpy as np


def get_inactive_board(gamestate):
    return gamestate._playfield.get_bool_board().astype(float)

def get_active_board(gamestate):
    playfield = Playfield()
    playfield.insert_piece(gamestate._active_piece, gamestate._active_piece.coords)
    return playfield.get_bool_board().astype(float)

def get_board_state(gamestate):
    return torch.stack((torch.tensor(get_inactive_board(gamestate)), torch.tensor(get_active_board(gamestate)))).float()

def get_piece(gamestate, piece):
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

def get_piece_state(gamestate, piece):
    result = torch.zeros(7).float()
    result[get_piece(gamestate, piece)] = 1.0
    return result

def get_next_piece(gamestate):
    return get_piece_state(gamestate, gamestate._next_piece)

def get_current_piece(gamestate):
    return get_piece_state(gamestate, gamestate._active_piece)

def get_raw_state(gamestate):
    board_state = get_inactive_board(gamestate)
    board_state = torch.tensor(board_state.reshape(200)).float()
    next_piece_state = get_next_piece(gamestate)
    current_piece_state = get_current_piece(gamestate)
    current_piece_coord = torch.tensor(gamestate._active_piece.coords).float()
    state = torch.cat([board_state, current_piece_coord, current_piece_state, next_piece_state])
    return state, board_state

def get_enclosed_space(gamestate):
    def get_open_space(board_state, x, y, visited):
        if (x,y) in visited or board_state[x,y] == 1.0:
            return 0
        visited.add((x,y))
        result = 1
        if x-1 >= 0:
            result += get_open_space(board_state, x-1, y, visited)
        if y-1 >= 0:
            result += get_open_space(board_state, x, y-1, visited)
        if x+1 < 10:
            result += get_open_space(board_state, x+1, y, visited)
        if y+1 < 20:
            result += get_open_space(board_state, x, y+1, visited)
        return result
    board_state = get_inactive_board(gamestate)
    visited = set()
    open_spaces = get_open_space(board_state, 5, 19, visited)
    total_spaces = np.sum(1.0 - board_state)
    return total_spaces - open_spaces


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(2,4,3,2,1)
        self.conv1b = nn.Conv2d(4,4,3,1,1)
        self.conv2 = nn.Conv2d(4,8,3,2,1)
        self.conv2b = nn.Conv2d(8,8,3,1,1)
        self.fc1 = nn.Linear(127,1000)
        self.fc2 = nn.Linear(1000,500)
        self.fc3 = nn.Linear(500, 4)
    def forward(self, board, piece):
        board = self.conv1(board)
        board = self.relu(board)
        board = self.conv1b(board)
        board = self.relu(board)
        board = self.conv2(board)
        board = self.relu(board)
        board = self.conv2b(board)
        board = self.relu(board)
        board = board.reshape(board.shape[0],-1)
        out = torch.cat([board,piece],1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

        
class Model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(216, 432)
        self.fc2 = nn.Linear(432, 862)
        self.fc3 = nn.Linear(862, 4)

    def forward(self, state):
        state = self.fc1(state)
        state = self.relu(state)
        state = self.fc2(state)
        state = self.relu(state)
        state = self.fc3(state)
        return state
