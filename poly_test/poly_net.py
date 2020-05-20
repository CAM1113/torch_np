import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt


class poly_net(nn.Module):
    def __init__(self):
        super(poly_net, self).__init__()
        self.net = nn.Sequential(

            nn.Linear(in_features=1, out_features=10, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.05),

            nn.Linear(in_features=10, out_features=100, bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.05),

            nn.Linear(in_features=100, out_features=500, bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.05),

            nn.Linear(in_features=500, out_features=100, bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.05),

            nn.Linear(in_features=100, out_features=10, bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.05),

            nn.Linear(in_features=10, out_features=1, bias=False),
        )

    def forward(self, x):
        return self.net(x)


def func(x):
    return x + 10 * np.sin(x * 5) + 7 * np.cos(4 * x)


def generator_examples(num=1000):
    x = np.random.random(size=num) * 20 - 10
    y = func(x)
    x = torch.from_numpy(x).view(-1, 1).float()
    y = torch.from_numpy(y).view(-1, 1).float()
    return x, y


if __name__ == '__main__':
    net = poly_net()
    criterion = nn.MSELoss()
    lr = 1e-3
    optimizer = optim.Adam(net.parameters(), lr=lr, )

    for i in range(10000):
        optimizer.zero_grad()
        x_data, y_data = generator_examples()
        x_data = Variable(x_data)
        y_data = Variable(y_data)
        y_pred = net(x_data)
        loss = criterion(y_data, y_pred)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(loss.item())
        if i % 500 == 0:
            lr = lr / 10
            weight = net.parameters()

    # with torch.no_grad():
    x_pre = torch.arange(start=-10, step=0.01, end=10).view(-1, 1)
    y_pre = net(x_pre)
    x_pre = x_pre.view(-1).detach().numpy()
    y_pre = y_pre.view(-1).detach().numpy()
    plt.scatter(x_pre, y_pre)
    y_true = func(x_pre)
    plt.plot(x_pre, y_true)
    plt.show()
