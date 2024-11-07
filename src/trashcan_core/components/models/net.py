from torch import nn

from abc import ABC, abstractmethod

from trashcan_core.components.trainer import Trainer


class Net(ABC, nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    def __call__(self, x):
        return self.forward(x)

    @abstractmethod
    def forward(self, x):
        pass

    def as_trainer(self, **kwargs):
        return Trainer(net=self, **kwargs)
