from gamestate import *
from playfield import *
from playfield_controller import *
from modeling import *
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import os
from PIL import Image
from PIL import ImageFont

gamma = 0.95
num_episode = 1000000000
pool_size = 10
#criterion = nn.CrossEntropyLoss(reduction='none')
model = ResNetWithoutBN()
model.apply(init_weights)
optim = torch.optim.Adam(model.parameters(), lr=0.0001)
batch_size = 65536

state_pool = []
action_pool = []
reward_pool = []

max_step = 10000
for e in range(num_episode):
    pc = PlayfieldController()
    pc.update()
    prev_score = 0
    prev_pieces = 0
    prev_closed_spaces = 0
    step = 0
    model.eval()
    while not pc._game_over and step < max_step:
        for i in range(5):
            gs = pc.gamestate()
            get_raw_state(gs)
            #board_state = get_board_state(gs).unsqueeze(0)
            #piece_state = get_piece_state(gs).unsqueeze(0)
            #prob_action = model(board_state, piece_state)
            state, board_state = get_raw_state(gs)
            state = state.unsqueeze(0)
            prob_action = model(state)
            action = Categorical(logits = prob_action).sample()
            if action.item() == 0:
                pc.rotate_cw()
            elif action.item() == 1:
                pc.move_left()
            elif action.item() == 2:
                pc.move_right()
            if i != 4:
                state_pool.append(state)
                action_pool.append(action)
                reward_pool.append(0)
        pc.update()
        pieces = torch.sum(board_state).item()
        change_pieces = pieces - prev_pieces
        closed_spaces = get_enclosed_space(gs)
        change_closed_spaces = closed_spaces - prev_closed_spaces
        reward = (pc._score - prev_score) * 10 - change_closed_spaces * 2 + change_pieces
        prev_pieces = pieces
        prev_score = pc._score
        prev_closed_spaces = closed_spaces
        step+=1
        if pc._game_over or step == max_step:
            reward = "game_over"
        state_pool.append(state)
        action_pool.append(action)
        reward_pool.append(reward)
        
        
    model.train()
    if e % pool_size == pool_size - 1:
        for i in reversed(range(len(reward_pool))):
            if reward_pool[i] == "game_over":
                prev_reward = 0
            else:
                prev_reward = prev_reward * gamma + reward_pool[i]
            reward_pool[i] = prev_reward
        reward_mean = np.mean(reward_pool)
        reward_std = np.std(reward_pool)
        print("%d games played. Current reward pool mean: %f. Number of States %d" % ((e+1), reward_mean, len(reward_pool)))
        if reward_std == 0.0:
            print("0.0 std. Skipped")
            continue
        
        for i in range(len(reward_pool)):
            reward_pool[i] = (reward_pool[i] - reward_mean) / reward_std
            

        optim.zero_grad()
        for i in range(0, len(reward_pool), batch_size):
            if reward_pool[i] == 0.0:
                continue
            #board_state, piece_state = state_pool[i]
            state = torch.cat(state_pool[i:min(len(state_pool),i+batch_size)])
            action = torch.tensor(action_pool[i:min(len(action_pool),i+batch_size)]).long()
            reward = torch.tensor(reward_pool[i:min(len(reward_pool),i+batch_size)]).float()
            loss = torch.sum( -Categorical(logits = model(state)).log_prob(action) * reward)
            loss.backward()
        optim.step()
        state_pool = []
        action_pool = []
        reward_pool = []

        torch.save(model, "model.pth")

