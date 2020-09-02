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


def f(node, n):
    c = 0
    return node.value + c * np.sqrt(n + 1) / (node.n + 1)

model = Model7().cuda()

pc = PlayfieldController()
pc.update()
tree = MCTS( pc=pc, gamma=0.95)
datasets, reward = tree.generate_a_game(num_iter=200)
"""
np.random.seed(1)

pc = PlayfieldController()
pc.update()
tree = MCTS(model=model, pc=pc, gamma=0.95)
print(tree.root.value)
print(tree.root.child_nodes)
tree.stimulate_to_leaf(tree.root, deepcopy(pc))
print(tree.root.child_nodes[0].value)
print(tree.root.value)
tree.stimulate_to_leaf(tree.root, deepcopy(pc))
tree.stimulate_to_leaf(tree.root, deepcopy(pc))
tree.stimulate_to_leaf(tree.root, deepcopy(pc))

for _ in range(8):
    tree.stimulate_to_leaf(tree.root, deepcopy(pc))
    values = []
    for node in tree.root.child_nodes:
        values.append(node.value)
    print(values)
    values = []
    for node in tree.root.child_nodes:
        values.append(f(node, tree.root.n))
    print(values)
    
tree.search(pc, 50)
values = []
for node in tree.root.child_nodes:
    values.append(node.value)
print(values)
values = []
for node in tree.root.child_nodes:
    values.append(f(node, tree.root.n))
print(values)
values = []
for node in tree.root.child_nodes:
        values.append(len(node.experiences))
print(values)

print(len(tree.make_move_remove_and_dump_unused_branch(0, pc)))

datasets = []
for i in range(1):
    print(i)
    pc = PlayfieldController()
    pc.update()
    tree = MCTS(model=model, pc=pc, gamma=0.95)
    datasets += tree.generate_a_game()[0]
"""

optim = torch.optim.Adam(model.parameters(), lr=0.001)
random.shuffle(datasets)
criterion1 = nn.SmoothL1Loss()
criterion = nn.SmoothL1Loss(reduction='sum')
valid_datasets = Gameplays(datasets[:int(len(datasets)*0.1)])
train_datasets = Gameplays(datasets[int(len(datasets)*0.1):])
train_loader = DataLoader(train_datasets, batch_size=1024)
valid_loader = DataLoader(valid_datasets, batch_size=1024)
result = []
average = 0.0
for e in range(1, 100):
    total_loss = 0
    for state, value in valid_loader:
        predicted_value = model(state.cuda())
        value = value.float().cuda()
        value = value.unsqueeze(1).cuda()
        loss = criterion(predicted_value, value)
        total_loss += loss.item()
    total_loss /= len(valid_datasets)
    print(total_loss)
    
    for state, value in train_loader:
        predicted_value = model(state.cuda())
        value = value.float().unsqueeze(1).cuda()
        value = value.unsqueeze(1).cuda()
        loss = criterion(predicted_value, value)
        optim.zero_grad()
        loss.backward()
        optim.step()
    

    average += total_loss
    if e % 5 == 0:
        result.append(average)
        average = 0.0
    
"""
pc._playfield.insert_piece(O((-1,-1)),(-1,-1))
pc._playfield.insert_piece(O((1,-1)),(1,-1))
pc._playfield.insert_piece(O((3,-1)),(3,-1))
pc._playfield.insert_piece(O((5,-1)),(5,-1))
pc._rng_queue = np.array([O])
pc.update()
pc.move_right()
pc.move_right()
pc.move_right()
pc.move_right()
pc.move_right()
pc.move_right()
pc.move_right()
for i in range(20):
    pc.update()
    gs = pc.gamestate()
    gs.plot()
while not pc.game_over:
    # num_moves = randint(5)
    num_moves = 1
    for i in range(num_moves):
        move = randint(0,4)
        if move == 0:
            pc.move_left()
            gs = pc.gamestate()
            gs.plot()
        elif move == 1:
            pc.move_right()
            gs = pc.gamestate()
            gs.plot()
        elif move == 1:
            pc.rotate_cw()
            gs = pc.gamestate()
            gs.plot()
        elif move == 3:
            pc.rotate_ccw()
            gs = pc.gamestate()
            gs.plot()
    pc.update()
    gs = pc.gamestate()
    gs.plot()
"""
