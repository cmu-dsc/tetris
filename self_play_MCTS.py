from playfield_controller import PlayfieldController
from gamestate import *
from playfield import *
from playfield_controller import *
from modeling import *
from torch.utils.data import DataLoader
import torch

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/MCTS-makedata-heuristics-50sims")

for i in range(1000000000):
    pc = PlayfieldController()
    pc.update()
    tree = MCTS_heuristics(pc=pc, gamma=0.95)
    data, actual_data, _ = tree.generate_a_game(num_iter=200, max_steps=500, stats_writer=(writer, i))

    states = []
    values = []
    for s, v in data:
        states.append(s)
        values.append(v)
    states = torch.stack(states)
    values = torch.tensor(values)
    torch.save(states, "storage_heuristics/data/states/%d.pt" % i)
    torch.save(values, "storage_heuristics/data/values/%d.pt" % i)

    states = []
    values = []
    actions = []
    for s, v, a in actual_data:
        states.append(s)
        values.append(v)
        actions.append(a)
    states = torch.stack(states)
    values = torch.tensor(values)
    actions = torch.tensor(actions)
    torch.save(states, "storage_heuristics/actual_data/states/%d.pt" % i)
    torch.save(values, "storage_heuristics/actual_data/values/%d.pt" % i)
    torch.save(actions, "storage_heuristics/actual_data/actions/%d.pt" % i)
    
