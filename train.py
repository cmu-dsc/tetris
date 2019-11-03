from gamestate import *
from playfield import *
from playfield_controller import *
from modeling import *
import torch
import torch.nn as nn
from torch.distributions import Categorical

gamma = 0.99
num_episode = 100
pool_size = 10
criterion = nn.BCEWithLogitsLoss()
model = Model()
optim = torch.optim.Adam(model.parameters(), lr=0.01)

state_pool = []
action_pool = []
reward_pool = []
for e in range(num_episode):
    pc = PlayfieldController()
    pc.update()
    prev_score = 0
    print(e)
    while True:
        gs = pc.gamestate()
        #gs.plot()
        board_state = get_board_state(gs).unsqueeze(0)
        piece_state = get_piece_state(gs).unsqueeze(0)
        prob_action = model(board_state, piece_state)
        action = Categorical(logits = prob_action).sample()
        action = 0
        print(action)
        if action == 0:
            pc.rotate_ccw()
        elif action == 1:
            pc.move_left()
        elif action == 2:
            pc.move_right()
        pc.update()
        reward = pc._score - prev_score
        prev_score = pc._score
        if pc._game_over:
            break
        
        
