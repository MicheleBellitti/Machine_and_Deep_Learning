import torch
import torch.nn as nn
import torchvision.datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import multiprocessing as mp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 10)
        self.opt = torch.optim.Adam(self.parameters(), lr=0.01)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def fit(self,train_loader, epochs=1000, verbose=True):
        for epoch in range(epochs):
            for data, labels in train_loader:
                data = data.view(-1, 28*28)
                data, labels = data.to(device), labels.to(device)
                self.opt.zero_grad()
                y_pred = self(data)
                l = self.loss(y_pred, labels)
                l.backward()
                self.opt.step()
            if verbose:
                if epoch % 10 == 0:
                    print(f'epoch: {epoch}/{epochs} ({round(100*epoch/epochs,1)}%) loss: {l.item()}')

    def test(self, test_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in test_loader:
                data = data.view(-1, 28*28)
                data, labels = data.to(device), labels.to(device)
                y_pred = self(data)
                _, predicted = torch.max(y_pred, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Accuracy: {100 * correct / total}%')


if __name__ == '__main__':
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    net = Net()
    net.to(device)
    net.fit(train_loader)
    net.test(test_loader)