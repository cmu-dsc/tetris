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
writer = SummaryWriter("runs/DDQN-0.999-model6-0.001")


gamma = 0.95
num_episode = 1000000000
model = Model6().cuda()
model_target = Model6().cuda()
optim = torch.optim.Adam(model.parameters(), lr=1e-4, eps=1e-3)
replay_size = 4194304
batch_size = 1024
epsilon = 1
criterion = nn.SmoothL1Loss()
eps_decay = 0.999

replay_buffer = ReplayMemory(replay_size)

max_step = 10000

counter = 0
for e in range(num_episode):
    pc = PlayfieldController()
    pc.update()
    prev_score = 0
    step = 0
    prev_suface_area = 0
    prev_closed_spaces = 0
    prev_max_height = 0
    
    total_rewards = []
    num_states = 0
    losses = []

    if e % 10 == 0:
        model_target.load_state_dict(model.state_dict())
    while not pc._game_over and step < max_step:
        model.eval()
        num_states += 1
        done = False
        gs = pc.gamestate()
        state = get_new_board_state(gs)
        if random.random() < epsilon:
            action = random.randint(0,3)
        else:
            q_value = model(state.unsqueeze(0).cuda())
            action = torch.argmax(q_value).item()
            
        if action == 0:
            pc.move_left()
        elif action == 1:
            pc.move_right()
        elif action == 2:
            pc.rotate_cw()

        pc.update()

        reward = pc._score - prev_score
        prev_score = pc._score

        bb = pc._playfield.get_bool_board()
        bb = np.rot90(bb, k = 1)
        sa = surface_area(bb)
        if sa < prev_suface_area:
            reward += 2
        prev_suface_area = sa
        

        max_height = 0
        for ind, row in enumerate(bb):
            if np.any(row):
                max_height = bb.shape[0] - ind - 1
                break
        reward += prev_max_height - max_height

        
        prev_max_height = max_height
        
        closed_spaces = get_enclosed_space(gs)
        change_closed_spaces = closed_spaces - prev_closed_spaces
        reward -= closed_spaces - prev_closed_spaces
        prev_closed_spaces = closed_spaces
        step+=1
        if pc._game_over or step == max_step:
            done = True

        total_rewards.append(reward)

        next_state = get_new_board_state(pc.gamestate())

        replay_buffer.push(state, action, reward, next_state, done)

        counter += 1
        if len(replay_buffer) >= batch_size and counter > 10:
            counter = 0
            model.train()
            s, a, r, ns, d = replay_buffer.sample(batch_size)
            predicted_Q = model(s.cuda()).gather(1, a.unsqueeze(1).cuda()).squeeze(1)
            target_Q = r.cuda() + gamma * model_target(ns.cuda()).gather(1, torch.max(model(ns.cuda()), dim=1)[1].unsqueeze(1)).squeeze(1) * (1-d).cuda()
            loss = criterion(predicted_Q, target_Q)
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.item())

    print("%d games played. Current reward: %f. Number of States %d" % ((e+1), np.sum(total_rewards), num_states))
    writer.add_scalar('Rewards', np.sum(total_rewards), e)
    writer.add_scalar('Number of states', num_states, e)
    writer.add_scalar('Loss mean', np.mean(losses), e)
    writer.add_scalar('Epsilon', epsilon, e)
    writer.flush()
    epsilon = max(epsilon * eps_decay, 0.1)
    torch.save(model, "DDQN-0.999-model6-0.001.pth")

