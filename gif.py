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

model = torch.load("model.pth")
pc = PlayfieldController()
pc.update()
count = 0
e = 0

os.mkdir('figs/')

while not pc._game_over:
    for i in range(5):
            gs = pc.gamestate()
            #board_state = get_board_state(gs).unsqueeze(0)
            #piece_state = get_piece_state(gs).unsqueeze(0)
            state, board_state = get_raw_state(gs)
            state = state.unsqueeze(0)
            prob_action = model(state)
            action = Categorical(logits = prob_action).sample()
            if action == 0:
                pc.rotate_cw()
            elif action == 1:
                pc.move_left()
            elif action == 2:
                pc.move_right()
            gs.plot('figs/im%d.jpg' % count)
            count += 1
    pc.update()
images = []

for c in range(count):
    im = Image.open("figs/im%d.jpg" % (c))
    images.append(im)
images[0].save("games/%d.gif" % e, save_all=True, append_images = images[1:],loop=0,duration=100)
os.remove('figs/')

