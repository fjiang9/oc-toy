import torch
from torch.utils.data import Dataset, DataLoader

class fashionMNIST(Dataset):
    def __init__(self, X, y, real=[0], fake=[1, 2, 3, 4, 5]):
        super(fashionMNIST, self).__init__()
        self.real = real
        self.fake = fake
        keep_ind = torch.zeros(y.shape[0])
        for target in real+fake:
            keep_ind += (y == target)
        keep_ind = keep_ind.bool()
        self.X = torch.from_numpy(X[keep_ind]).unsqueeze(1).float()
        self.y = torch.from_numpy(y[keep_ind])


    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        input_feature = self.X[idx]
        label = self.y[idx]
        if label in self.real:
            label = 0
        else:
            label = 1

        return input_feature, label

