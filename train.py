from gamestate import *
from playfield import *
from playfield_controller import *
from modeling import *
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import os

gamma = 0.95
num_episode = 1000
pool_size = 10
criterion = nn.CrossEntropyLoss()
model = Model()
optim = torch.optim.Adam(model.parameters(), lr=0.001)

state_pool = []
action_pool = []
reward_pool = []
for e in range(num_episode):
    pc = PlayfieldController()
    pc.update()
    prev_score = 0
    count = 0
    while True:
        gs = pc.gamestate()
        board_state = get_board_state(gs).unsqueeze(0)
        piece_state = get_piece_state(gs).unsqueeze(0)
        prob_action = model(board_state, piece_state)
        action = Categorical(logits = prob_action).sample()
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
            reward = -100
        state_pool.append((board_state, piece_state))
        action_pool.append(action)
        reward_pool.append(reward)
        if e % pool_size == pool_size - 1:
            gs.plot('figs/im%d.png' % count)
        count += 1
        if pc._game_over:
            break

    if e % pool_size == pool_size - 1:
        for i in reversed(range(len(reward_pool))):
            if reward_pool[i] == -100:
                prev_reward = -100
            else:
                prev_reward = prev_reward * gamma + reward_pool[i]
                reward_pool[i] = prev_reward
        reward_mean = np.mean(reward_pool)
        reward_std = np.std(reward_pool)
        print(reward_mean, flush=True)
        for i in range(len(reward_pool)):
            reward_pool[i] = (reward_pool[i] - reward_mean) / reward_std
            

        optim.zero_grad()
        for i in range(len(reward_pool)):
            board_state, piece_state = state_pool[i]
            action = torch.tensor([action_pool[i]]).long()
            reward = torch.tensor(reward_pool[i]).unsqueeze(0)
            loss = criterion(model(board_state, piece_state), action) * reward
            loss.backward()
        optim.step()
        state_pool = []
        action_pool = []
        reward_pool = []
        os.system('convert -delay 5 -loop 0 %s %d.gif' % (''.join(['figs/im%d.png ' % i for i in range(count)]), e))
        os.system('rm figs/*')