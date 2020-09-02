from torch.utils.data.dataset import Dataset
import torch
from modeling import *
from torch.utils.data import DataLoader


class Gameplays(Dataset):
    def __init__(self, index):
        self.state = torch.load("storage/data/states/%d.pt" % index)
        self.value = torch.load("storage/data/values/%d.pt"% index)
    def __len__(self):
        return len(self.state)
    def __getitem__(self, index):
        feature = self.state[index]
        label = self.value[index]
        return feature, label


model = Model7().cuda()
criterion = nn.SmoothL1Loss(reduction='sum')
optim = torch.optim.Adam(model.parameters(), lr=0.001)

while True:
    valid_datasets = Gameplays(0)
    valid_loader = DataLoader(valid_datasets, batch_size=1024)
    total_loss = 0
    for state, value in valid_loader:
        predicted_value = model(state.cuda())
        value = value.float().cuda()
        value = value.unsqueeze(1).cuda()
        loss = criterion(predicted_value, value)
        total_loss += loss.item()
    total_loss /= len(valid_datasets)
    print(total_loss)
    for i in range(1,78):
        train_datasets = Gameplays(i)
        train_loader = DataLoader(train_datasets, batch_size=1024)
        for state, value in train_loader:
            predicted_value = model(state.cuda())
            value = value.float().unsqueeze(1).cuda()
            loss = criterion(predicted_value, value)
            optim.zero_grad()
            loss.backward()
            optim.step()
    torch.save(model, "value_net_sigmoid_v2.pth")
    
