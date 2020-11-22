from playfield_controller import PlayfieldController
from gamestate import *
from playfield import *
from playfield_controller import *
import numpy.random
from numpy.random import randint
from modeling import *
from copy import deepcopy
from torch.utils.data import DataLoader
import random

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/test-vanillaMCTS")


# Testing for vanillas MCTS
game = 0
num_iter=50
while True:
    average_reward = 0
    for _ in range(10):
        pc = PlayfieldController()
        pc.update()
        tree = MCTS(model=None, pc=pc, gamma=0.95)
        _, reward, steps, action_stats = tree.generate_a_game(num_iter=num_iter)

        writer.add_scalar('Rewards', reward, game)
        writer.add_scalar('Steps', steps, game)
        writer.add_scalar('Move Left', action_stats[0], game)
        writer.add_scalar('Move Right', action_stats[1], game)
        writer.add_scalar('Rotate', action_stats[2], game)
        writer.add_scalar('Nothing', action_stats[3], game)
        writer.add_scalar('Iterations', num_iter, game)
        game+= 1
        average_reward += reward
        #print("%d game, %f rewards, %d num of steps" % (game, reward, steps))
    average_reward /= 10
    print("%d iterations, %f average reward" % (num_iter, average_reward))
    num_iter += 50    
        
