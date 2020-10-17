from torch.utils.data.dataset import Dataset
import torch
from modeling import *
from torch.utils.data import DataLoader
from os import path

class Gameplays(Dataset):
    def __init__(self, index):
        self.state = torch.load("storage_heuristics/data/states/%d.pt" % index)
        self.value = torch.load("storage_heuristics/data/values/%d.pt"% index)
    def __len__(self):
        return len(self.state)
    def __getitem__(self, index):
        feature = self.state[index]
        label = self.value[index]
        return feature, label


model = Model7().cuda()
#criterion = nn.MSELoss(reduction='mean')
#criterion2 = nn.BCELoss(reduction='mean')
optim = torch.optim.Adam(model.parameters())
e = 0
while True:
    """
    total_loss = 0
    counter = 0
    for i in range(0, 5):
        if not path.exists("storage_heuristics/data/values/%d.pt" % i ):
            continue
        valid_datasets = Gameplays(i)
        valid_loader = DataLoader(valid_datasets, batch_size=1024)
        for state, value in valid_loader:
            predicted_value = model(state.cuda())
            value = value.unsqueeze(1).float().cuda()
            loss = criterion(predicted_value, value)
            total_loss += loss.item()
            counter += 1
    total_loss /= counter
    print("average valid loss %f on iteration %d" % (total_loss, e))
    """
    
    e += 1
    for i in range(5,288):
        if not path.exists("storage_heuristics/data/values/%d.pt" % i ):
            continue
        train_datasets = Gameplays(i)
        train_loader = DataLoader(train_datasets, batch_size=256)
        for state, value in train_loader:
            predicted_value = model(state.cuda())
            value = torch.min(value.float(), torch.tensor([1.0])).unsqueeze(1).cuda()
            #value = torch.bernoulli(value)
            #loss = criterion2(predicted_value, value)
            loss = - torch.mean( value * predicted_value.log() + (1.0 - value) * (1 - predicted_value).log())
            print(loss)
            optim.zero_grad()
            loss.backward()

            #total_norm = 0
            #for p in model.parameters():
            #    param_norm = p.grad.data.norm(2)
            #    total_norm += param_norm.item() ** 2
            #total_norm = total_norm ** (1. / 2)
            
            optim.step()
    torch.save(model, "pretrained_value_nets3/%d.pth" % e)

    
    
