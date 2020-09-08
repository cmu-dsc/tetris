from playfield_controller import PlayfieldController
from gamestate import *
from playfield import *
from playfield_controller import *
from modeling import *
from torch.utils.data import DataLoader
import torch

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/MCTS-makedata-500sims")

while True:
    with open("i", "r+") as f:
        i = int(f.read())
        f.seek(0)
        f.write(str(i+1))
    pc = PlayfieldController()
    pc.update()
    tree = MCTS(pc=pc, gamma=0.95)
    data, actual_data, _ = tree.generate_a_game(num_iter=500, max_steps=750, stats_writer=(writer, i))

    states = []
    values = []
    for s, v, _ in data:
        states.append(s)
        values.append(v)
    states = torch.stack(states)
    values = torch.tensor(values)
    torch.save(states, "storage2/data/states/%d.pt" % i)
    torch.save(values, "storage2/data/values/%d.pt" % i)

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
    torch.save(states, "storage2/actual_data/states/%d.pt" % i)
    torch.save(values, "storage2/actual_data/values/%d.pt" % i)
    torch.save(actions, "storage2/actual_data/actions/%d.pt" % i)
    
