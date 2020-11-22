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

model = torch.load("DDQN-0.999-model6-0.001.pth").cpu()
model.eval()
pc = PlayfieldController()
pc.update()
count = 0
while not pc._game_over:
    gs = pc.gamestate()
    state = get_new_board_state(gs)
    state = state.unsqueeze(0)
    prob_action = model(state)
    action = torch.argmax(prob_action)
    if action == 0:
        pc.move_left()
    elif action == 1:
        pc.move_right()
    elif action == 2:
        pc.rotate_cw()
        
    gs.plot('figs/im%d.jpg' % count)
    count += 1
    pc.update()
    if pc._game_over:
        gs.plot()
images = []

for c in range(count):
    im = Image.open("figs/im%d.jpg" % (c))
    images.append(im)
images[0].save("games/%s.gif" % "DDQN-0.999-model6-0.001", save_all=True, append_images = images[1:],loop=0,duration=1)
