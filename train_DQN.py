from gamestate import *
from playfield import *
from playfield_controller import *
from modeling import *
import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
from PIL import ImageFont
import random
from surface_area import surface_area

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


gamma = 0.95
num_episode = 1000000000
pool_size = 10
model = ANN(216, 8, [432, 862, 862*2])
optim = torch.optim.Adam(model.parameters(), lr=0.0005)
replay_size = 65536
batch_size = 1024
epsilon = 0.3
criterion = nn.SmoothL1Loss()
eps_decay = 0.99

replay_buffer = ReplayMemory(replay_size)

max_step = 10000
for e in range(num_episode):
    pc = PlayfieldController()
    pc.update()
    prev_score = 0
    prev_pieces = 0
    prev_max_height = 0
    prev_suface_area = 0
    step = 0

    total_rewards = []
    num_states = 0
    while not pc._game_over and step < max_step:
        model.eval()
        num_states += 1
        done = False
        gs = pc.gamestate()
        state, board_state = get_raw_state(gs)
        if random.random() < epsilon:
            action = random.randint(0,7)
        else:
            q_value = model(state.unsqueeze(0))
            action = torch.argmax(q_value).item()
            
        if action == 0:
            for zzz in range(3):
                pc.move_left()
        elif action == 1:
            for zzz in range(2):
                pc.move_left()
        elif action == 2:
            pc.move_left()
        elif action == 3:
            pc.move_right()
        elif action == 4:
            for zzz in range(2):
                pc.move_right()
        elif action == 5:
            for zzz in range(3):
                pc.move_right()
        elif action == 6:
            pc.rotate_cw()

        pc.update()
        pieces = torch.sum(board_state).item()
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
            reward = 0
            done = True

        total_rewards.append(reward)

        next_state = get_raw_state(pc.gamestate())[0]

        replay_buffer.push(state, action, reward, next_state, done)
        
        if len(replay_buffer) >= batch_size:
            model.train()
            s, a, r, ns, d = replay_buffer.sample(batch_size)
            predicted_Q = model(s).gather(1, a.unsqueeze(1)).squeeze(1)
            target_Q = r + gamma * torch.max(model(ns), dim=1)[0] * (1-d)
            loss = criterion(predicted_Q, target_Q)
            optim.zero_grad()
            loss.backward()
            optim.step()

    print("%d games played. Current reward pool mean: %f. Number of States %d" % ((e+1), np.mean(total_rewards), num_states))
    writer.add_scalar('Reward mean', np.mean(total_rewards), e)
    writer.add_scalar('Number of states', num_states, e)
    writer.flush()
    epsilon = max(epsilon * eps_decay, 0.01)
    torch.save(model, "DQN_2.pth")

