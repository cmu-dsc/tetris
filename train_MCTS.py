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

def train_dataset(datasets):
    global game
    global writer
    global model
    global model_temp
    
    model.train()
    total_training_loss = 0.0
    optim = torch.optim.Adam(model.parameters())
    total_valid_data = []
    for data in datasets[:3]:
        total_valid_data += data
    valid_datasets = Gameplays(total_valid_data)
    total_data = []
    for data in datasets[3:]:
        total_data += data
    train_datasets = Gameplays(total_data)
    train_loader = DataLoader(train_datasets, batch_size=64)
    valid_loader = DataLoader(valid_datasets, batch_size=64)
    
    prev_valid_loss_average = None
    c = 0
    average_norm = 0.0
    counter = 0

    while c < 200:
        temp_total_training_loss = 0.0
        temp_average_norm = 0.0

        total_loss = 0
        for state, value in valid_loader:
            predicted_value = model(state)
            value = torch.min(value.float(), torch.tensor([1.0]))
            loss = - torch.mean( value * predicted_value.log() + (1.0 - value) * (1 - predicted_value).log())
            total_loss += loss.item()
        total_loss /= len(valid_loader)

        print(total_loss)
        
        valid_loss_average = total_loss
        c += 1
        if prev_valid_loss_average == None or prev_valid_loss_average > valid_loss_average:
            prev_valid_loss_average = valid_loss_average
            model_temp.load_state_dict(model.state_dict())
            counter = 0
        else:
            counter += 1
        
        for state, value in train_loader:
            predicted_value = model(state)
            value = torch.min(value.float(), torch.tensor([1.0]))
            loss = - torch.mean( value * predicted_value.log() + (1.0 - value) * (1 - predicted_value).log())
            optim.zero_grad()
            loss.backward()
            total_norm = 0
            for p in model.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            optim.step()
            temp_total_training_loss += loss.item()
            
            temp_average_norm += total_norm
            
        average_norm = temp_average_norm / len(train_loader)
        total_training_loss = temp_total_training_loss / len(train_loader)
            
        
        if counter > 5:
            model.load_state_dict(model_temp.state_dict())
            torch.save(model, "MCTS.pth")
            break
    writer.add_scalar('Training loss', total_training_loss, game)
    writer.add_scalar('Training Norm', average_norm, game)
    writer.add_scalar('Validation loss', prev_valid_loss_average, game)
    writer.add_scalar('Number of Epoch', c, game)


writer = SummaryWriter("runs/MCTS3")

model = Model()
model_temp = Model()
game = 0
datasets = []
while True:
    pc = PlayfieldController()
    pc.update()
    model.eval()
    tree = MCTS(model=model, pc=pc, gamma=0.999)
    data, _, reward = tree.generate_a_game(num_iter=50, max_steps=500, stats_writer=(writer, game))
    game += 1
    datasets.append(data)
    if game % 25 == 0:
        train_dataset(datasets)
        datasets = []
        if game % 100 == 0:
            torch.save(model, "MCTS_%d.pth" % game)
        
