import torch.nn as nn


# How to apply exponential moving average decay for variables?
# https://discuss.pytorch.org/t/how-to-apply-exponential-moving-average-decay-for-variables/10856/2
class EMA(nn.Module):
    def __init__(self, mu):
        super().__init__()
        self.mu = mu

    def forward(self, current, last_average):
        if last_average is None:
            return current
        new_average = self.mu * current + (1 - self.mu) * last_average
        return new_average


class AVG:
    def __init__(self):
        self.sum = 0
        self.counter = 0

    def reset(self):
        self.__init__()

    def add_value(self, value):
        self.counter += 1
        self.sum += value

    def get_avg(self):
        return self.sum / self.counter

# ema = EMA(0.999)
# current = Variable(torch.rand(5),requires_grad=True)
# average = Variable(torch.zeros(5),requires_grad=True)
# average = ema(current, average)


def register(mf):
    mf.register_event('EMA', EMA, unique=True)
    mf.register_event('AVG', AVG, unique=True)
