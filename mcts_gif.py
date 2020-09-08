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
import shutil

model = torch.load("MCTS_value_soft_ce.pth").cuda()
model.eval()
#with open("highest","r") as f:
#    highest_reward = int(f.read())

highest_reward = 0
while True:
    os.mkdir("figs")
    pc = PlayfieldController()
    pc.update()
    count = 0
    tree = MCTS(model=model, pc=pc, gamma=0.95)
    total_reward = 0
    steps = 0

    while not pc._game_over and steps < 500:
        steps += 1
        gs = pc.gamestate()
        action = tree.search(pc, num_iter=50)
        if action == 0:
            pc.move_left()
        elif action == 1:
            pc.move_right()
        elif action == 2:
            pc.rotate_cw()
        tree.root = tree.root.child_nodes[action]
            
        gs.plot('figs/im%d.jpg' % count)
        count += 1
        prev_score = pc._score
        pc.update()
        reward = pc._score - prev_score
        total_reward += reward

    print(total_reward)
    if total_reward > highest_reward:
        images = []
        for c in range(count):
            im = Image.open("figs/im%d.jpg" % (c))
            images.append(im)
        images[0].save("games/%s.gif" % "MCTS_value_soft_ce", save_all=True, append_images = images[1:],loop=0,duration=1)
        with open("highest","w") as f:
            f.write(str(total_reward))
    shutil.rmtree("figs")

