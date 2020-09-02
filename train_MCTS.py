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
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    random.shuffle(datasets)
    criterion = nn.SmoothL1Loss(reduction='sum')
    valid_datasets = Gameplays(datasets[:int(len(datasets)*0.1)])
    train_datasets = Gameplays(datasets[int(len(datasets)*0.1):])
    train_loader = DataLoader(train_datasets, batch_size=1024)
    valid_loader = DataLoader(valid_datasets, batch_size=1024)
    
    valid_loss_average = 0
    prev_valid_loss_average = None
    c = 0
    while True:
        for state, value in train_loader:
            predicted_value = model(state.cuda())
            value = value.float().unsqueeze(1).cuda()
            loss = criterion(predicted_value, value)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_training_loss += loss.item()
            
        total_loss = 0
        for state, value in valid_loader:
            predicted_value = model(state.cuda())
            value = value.float().cuda()
            value = value.unsqueeze(1).cuda()
            loss = criterion(predicted_value, value)
            total_loss += loss.item()
        total_loss /= len(valid_datasets)
        
        valid_loss_average += total_loss
        c += 1

        if c % 5 == 0:
            if prev_valid_loss_average == None or prev_valid_loss_average > valid_loss_average:
                prev_valid_loss_average = valid_loss_average
                model_temp.load_state_dict(model.state_dict())
                valid_loss_average = 0
            else:
                model.load_state_dict(model_temp.state_dict())
                torch.save(model, "MCTS.pth")
                break
    writer.add_scalar('Total Training loss', total_training_loss, game)
    writer.add_scalar('Validation loss', prev_valid_loss_average, game)
    writer.add_scalar('Number of Epoch', c, game)


writer = SummaryWriter("runs/MCTS")

model = Model7().cuda()
model_temp = Model7().cuda()
datasets = []
reward_history = []
game = 0

while True:
    if game % 10 == 0:
        if len(datasets) != 0:
            train_dataset(datasets)
            datasets = []
        if len(reward_history) == 0 or np.mean(reward_history) < 40:
            model.eval()
            pc = PlayfieldController()
            pc.update()
            tree = MCTS(pc=pc, gamma=0.95)
            data, _, _ = tree.generate_a_game(num_iter=200, stats_writer=(writer, game))
            train_dataset(data)
            game += 1
        reward_history = []

    pc = PlayfieldController()
    pc.update()
    model.eval()
    tree = MCTS(model=model, pc=pc, gamma=0.95)
    data, _, reward = tree.generate_a_game(num_iter=200, stats_writer=(writer, game))
    datasets += data
    reward_history.append(reward)
    game += 1
    
        
