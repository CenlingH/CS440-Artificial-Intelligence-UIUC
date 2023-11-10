# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by James Soole for the Fall 2023 semester

"""
This is the main entry point for MP9. You should only modify code within this file.
The unrevised staff files will be used for all other files and classes when code is run, 
so be careful to not modify anything else.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from utils import get_dataset_from_arrays
from torch.utils.data import DataLoader


class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param yhat - an (N,out_size) Tensor
            @param y - an (N,) Tensor
            @return l(x,y) an () Tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension

        For Part 1 the network should have the following architecture (in terms of hidden units):
        in_size -> h -> out_size , where  1 <= h <= 256

        We recommend setting lrate to 0.01 for part 1.

        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        self.in_size = in_size
        self.out_size = out_size
        self.lrate = lrate
        self.func = nn.Sequential(
            nn.Linear(in_size, 64),
            nn.Sigmoid(),
            nn.Linear(64, out_size)
        )
        # 创建一个随机梯度下降的实例
        self.optimizer = optim.SGD(
            self.parameters(), self.lrate, momentum=0.9)

    def forward(self, x):
        """
        Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        return self.func(x)

    def step(self, x, y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) for this batch as a float (scalar)
        """
        self.optimizer.zero_grad()
        yhat = self.forward(x)
        loss = self.loss_fn(yhat, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()


def fit(train_set, train_labels, dev_set, epochs, batch_size=100):
    """ 
    Make NeuralNet object 'net'. Use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param epochs: an int, the number of epochs of training
    @param batch_size: size of each batch to train on. (default 100)

    This method *must* work for arbitrary M and N.

    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values in case your first choice does not seem to work well.

    @return losses: list of floats containing the total loss at the beginning and after each epoch.
        Ensure that len(losses) == epochs.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """
    net = NeuralNet(0.01, nn.CrossEntropyLoss(),
                    train_set.shape[-1], train_labels.max()+1)
    mean = train_set.mean()
    std = train_set.std()
    train_set = (train_set-mean)/std
    data = get_dataset_from_arrays(train_set, train_labels)
    losses = []
    for i in range(epochs):
        loss = 0
        for batch in DataLoader(data, batch_size=batch_size, shuffle=False):
            X, Y = batch['features'], batch['labels']
            loss = net.step(X, Y)
        losses.append(loss)
    mean = dev_set.mean()
    std = dev_set.std()
    dev_set = (dev_set - mean) / std  # 在没有梯度的情况下执行以下操作
    with torch.no_grad():
        yhats = net(dev_set).argmax(dim=-1).detach().numpy()
    return losses, yhats, net
