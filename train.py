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

from surface_area import surface_area
gamma = 0.95
num_episode = 1000000000
pool_size = 10
criterion = nn.CrossEntropyLoss(reduction='none')
# model = ResNet()
model = torch.load('model2.pth')
# model = Model3()
# model = Model2()
optim = torch.optim.Adam(model.parameters(), lr=0.0005)
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
    # prev_closed_spaces = 0
    prev_max_height = 0
    prev_suface_area = 0
    step = 0
    model.eval()
    while not pc._game_over and step < max_step:
        # for i in range(5):
        gs = pc.gamestate()
        get_raw_state(gs)
        #board_state = get_board_state(gs).unsqueeze(0)
        #piece_state = get_piece_state(gs).unsqueeze(0)
        #prob_action = model(board_state, piece_state)
        state, board_state = get_raw_state(gs)
        state = state.unsqueeze(0)
        prob_action = model(state)
        action = Categorical(logits = prob_action).sample()
        # if action.item() == 0:
        #     pc.rotate_cw()
        if action.item() == 0:
            for zzz in range(3):
                pc.move_left()
        elif action.item() == 1:
            for zzz in range(2):
                pc.move_left()
        elif action.item() == 2:
            pc.move_left()
        elif action.item() == 3:
            pc.move_right()
        elif action.item() == 4:
            for zzz in range(2):
                pc.move_right()
        elif action.item() == 5:
            for zzz in range(3):
                pc.move_right()
        # if i != 4:
        #     state_pool.append(state)
        #     action_pool.append(action)
        #     reward_pool.append(0)
        pc.update()
        pieces = torch.sum(board_state).item()
        # change_pieces = pieces - prev_pieces
        # closed_spaces = get_enclosed_space(gs)
        # change_closed_spaces = closed_spaces - prev_closed_spaces
        # reward = (pc._score - prev_score) * 10 - change_closed_spaces * 2 + change_pieces
        # prev_pieces = pieces
        # prev_score = pc._score
        # prev_closed_spaces = closed_spaces
        bb = pc._playfield.get_bool_board()
        bb = np.rot90(bb, k = 1)
        max_height = 0
        for ind, row in enumerate(bb):
            if np.any(row):
                max_height = bb.shape[0] - ind - 1
                break
        sa = surface_area(bb)
        reward = (prev_max_height - max_height)
        if pieces > prev_pieces and max_height == prev_max_height:
            reward += 1
        if sa <= prev_suface_area:
            reward += 2
        prev_pieces = pieces
        prev_max_height = max_height
        prev_suface_area = sa
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
            loss = torch.sum(criterion(model(state), action) * reward)
            loss.backward()
        optim.step()
        state_pool = []
        action_pool = []
        reward_pool = []

        # torch.save(model, "modelres.pth")
        torch.save(model, "model2.pth")

