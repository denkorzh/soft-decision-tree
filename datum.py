import os
import torch
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import Dataset

try:
    os.makedirs('./data')
except:
    print('directory ./data already exists')


class PravoDatasetTrain(Dataset):
    def __init__(self):
        super(PravoDatasetTrain, self).__init__()
        x = pd.read_csv('./pravo_data/nd_x_train_short.csv').values
        y = pd.read_csv('./pravo_data/nd_y_train.csv').values

        ratio = y.mean()
        self.weights = [ratio if el == 0 else 1-ratio for el in y]

        self.len = x.shape[0]
        self.x_data = torch.from_numpy(x).type(torch.FloatTensor)
        self.y_data = torch.from_numpy(y).type(torch.LongTensor).view(-1)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class PravoDatasetTest(Dataset):
    def __init__(self):
        super(PravoDatasetTest, self).__init__()
        x = pd.read_csv('./pravo_data/nd_x_test_short.csv').values
        y = pd.read_csv('./pravo_data/nd_y_test.csv').values
        self.len = x.shape[0]
        self.x_data = torch.from_numpy(x).type(torch.FloatTensor)
        self.y_data = torch.from_numpy(y).type(torch.LongTensor).view(-1)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


trainset = datasets.MNIST('./data', train=True,
                          download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))
                          ])
                          )

testset = datasets.MNIST('./data', train=False,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))
                         ])
                         )

# trainset = PravoDatasetTrain()
# testset = PravoDatasetTest()
