import torch
import torchvision.datasets
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import pandas as pd

data_train = pd.read_csv("./lab1_dataset/X_train.csv").values
label_train = pd.read_csv("./lab1_dataset/Y_train.csv").values
data_test = pd.read_csv("./lab1_dataset/X_test.csv").values
label_test = pd.read_csv("./lab1_dataset/Y_test.csv").values


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]

        return x, y


train = CustomDataset(data_train, label_train)
x_train = torch.utils.data.DataLoader(dataset=train, batch_size=400, shuffle=True)
test = CustomDataset(data_test, label_test)
x_test = torch.utils.data.DataLoader(dataset=test, batch_size=400, shuffle=True)


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()

        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.acv = torch.nn.Sigmoid()
        self.linear2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.acv(x)
        x = self.linear2(x)
        x = self.acv(x)
        return x


def training(model):
    cost = torch.nn.BCELoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.0001)
    epochs = 20000
    losses = []
    for i in range(epochs):
        sumloss = 0
        for data in x_train:
            x, y = data
            out = model(x)
            opt.zero_grad()
            loss = cost(out, y)
            loss.backward()
            opt.step()
            sumloss += loss
        if i % 500 == 0:
            losses.append(sumloss.item())

    return losses


def tes_ting(model):
    model.eval()
    for dt in x_train:
        x, y = dt
        pred = model(x)
        _, id = torch.max(pred.data, 1)
        pred = torch.round(pred.data.flatten())
        y = y.flatten()
        acc = torch.sum(pred == y).item()
        print(f'train accuracy = {acc / len(y)}')
    for data in x_test:
        x, y = data
        pred = model(x)
        _, id = torch.max(pred.data, 1)
        pred = torch.round(pred.data.flatten())
        y = y.flatten()
        acc = torch.sum(pred == y).item()
        print(f'test accuracy = {acc / len(y)}')


start = time.time()
mlp = MLP(29, 10, 1)
# mlp.cuda()
losses = training(mlp)
plt.plot(losses, marker='o', linestyle='-')
plt.show()


tes_ting(mlp)
end = time.time()
print("time cost: {} seconds".format(end - start))
print(losses)
