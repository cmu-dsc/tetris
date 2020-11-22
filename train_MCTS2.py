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
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

def train_dataset(datasets):
    global game
    global writer
    global model
    # global model_temp
    global total_epochs
    global optim
    global scheduler
    global train_loss_store
    global valid_loss_store

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.train()
    total_training_loss = 0.0
    total_valid_data = []
    for data in datasets[:16]:
        total_valid_data += data
    valid_datasets = Gameplays(total_valid_data)
    total_data = []
    for data in datasets[16:]:
        total_data += data
    train_datasets = Gameplays(total_data)
    train_loader = DataLoader(train_datasets, batch_size=512)
    valid_loader = DataLoader(valid_datasets, batch_size=512)
    
    valid_loss_average = 0
    c = 0
    average_norm = 0.0
    counter = 0

    while c < 25:
        temp_total_training_loss = 0.0
        temp_average_norm = 0.0

        total_loss = 0
        for state, value in valid_loader:
            state = state.to(device)
            value = value.to(device)
            predicted_value = model(state)
            value = torch.min(value.float(), torch.tensor([1.0]).to(device))
            loss = - torch.mean( value * predicted_value.log() + (1.0 - value) * (1 - predicted_value).log())
            total_loss += loss.item()
        total_loss /= len(valid_loader)

        print("Validation loss at epoch %d: %f" % (total_epochs, total_loss))
        valid_loss_store.append(total_loss)
        
        valid_loss_average = total_loss
        c += 1
        total_epochs += 1
        # if prev_valid_loss_average == None or prev_valid_loss_average > valid_loss_average:
        #     prev_valid_loss_average = valid_loss_average
        #     model_temp.load_state_dict(model.state_dict())
        #     counter = 0
        # else:
        #     counter += 1
        
        for state, value in train_loader:
            state = state.to(device)
            value = value.to(device)
            predicted_value = model(state)
            value = torch.min(value.float(), torch.tensor([1.0]).to(device))
            loss = - torch.mean( value * predicted_value.log() + (1.0 - value) * (1 - predicted_value).log())
            optim.zero_grad()
            loss.backward()
            # total_norm = 0
            # for p in model.parameters():
            #     param_norm = p.grad.data.norm(2)
            #     total_norm += param_norm.item() ** 2
            # total_norm = total_norm ** (1. / 2)
            optim.step()
            scheduler.step()
            temp_total_training_loss += loss.item()
            
            # temp_average_norm += total_norm
            
        # average_norm = temp_average_norm / len(train_loader)
        total_training_loss = temp_total_training_loss / len(train_loader)
        print("Training loss at epoch %d: %f" % (total_epochs, temp_total_training_loss/len(train_loader)))
        train_loss_store.append(temp_total_training_loss/len(train_loader))

        # if counter > 5:
        #     model.load_state_dict(model_temp.state_dict())
        #     torch.save(model, "MCTS.pth")
        #     break
        writer.add_scalar('Training loss', total_training_loss, total_epochs)
        # writer.add_scalar('Training Norm', average_norm, total_epochs)
        writer.add_scalar('Validation loss', valid_loss_average, total_epochs)
        # writer.add_scalar('Number of Epoch', c, total_epochs)



run_num = 11


# model = torch.load("results/8/MCTS_4608.pth")
model = Model()
# model_temp = Model()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)
# model_temp.to(device)
game = 0
datasets = []
lr = 0.001
LR_DROP_MILESTONES = [400,1000]
total_epochs = 0
optim = torch.optim.SGD(model.parameters(), lr = lr, momentum=0.9, weight_decay=0.000001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, LR_DROP_MILESTONES)
reward_store = []
train_loss_store = []
average_reward = []
valid_loss_store = []
writer = SummaryWriter("runs/%d" % run_num)
reward_total = 0
while total_epochs < 2000:
    pc = PlayfieldController()
    pc.update()
    model.eval()
    tree = MCTS(model=model, pc=pc, gamma=0.999)
    data, _, reward = tree.generate_a_game(num_iter=50, max_steps=500, stats_writer=(writer, game))
    game += 1
    datasets.append(data)
    reward_store.append(reward)
    reward_total+=reward
    if game % 144 == 0:
        average_reward.append(reward_total/144)
        writer.add_scalar("Average reward", reward_total/144, total_epochs)
        reward_total = 0
        train_dataset(datasets)
        datasets = []
        torch.save(model, "results/%d/MCTS_%d.pth" % (run_num,game))

        f = open("results/%d/log.txt" % run_num, "w")
        f.write("rewards: \n")
        f.write(str(reward_store))
        f.write("train loss: \n")
        f.write(str(train_loss_store))
        f.write("valid loss: \n")
        f.write(str(valid_loss_store))
        f.close()

torch.save(model, "results/%d/final" % (run_num))


