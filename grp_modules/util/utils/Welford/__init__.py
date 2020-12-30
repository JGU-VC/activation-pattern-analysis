# adaption of https://gist.github.com/alexalemi/2151722
#   by da-h
#       minor changes

import numpy as np


class Welford:
    """ Implements Welford's algorithm for computing a running mean
    and standard deviation as described at:
        http://www.johndcook.com/standard_deviation.html
    can take single values or iterables
    Properties:
        mean    - returns the mean
        std     - returns the std
        meanfull- returns the mean and std of the mean
    Usage:
        >>> foo = Welford()
        >>> foo(range(100))
        >>> foo
        <Welford: 49.5 +- 29.0114919759>
        >>> foo([1]*1000)
        >>> foo
        <Welford: 5.40909090909 +- 16.4437417146>
        >>> foo.mean
        5.409090909090906
        >>> foo.std
        16.44374171455467
        >>> foo.meanfull
        (5.409090909090906, 0.4957974674244838)
    """

    def __init__(self, lst=None):
        self.count = 0
        self.M = 0
        self.M2 = 0
        self.reset()
        self.__call__(lst)

    def update(self, x):
        if x is None:
            return
        self.count += 1
        delta = x - self.M
        self.M += delta / self.count
        delta2 = x - self.M
        self.M2 += delta * delta2

    def updateWithMeanVar(self, count, mean, var):
        new_count = self.count + count
        new_M = (self.M * self.count + mean * count) / new_count
        delta = mean - self.M
        self.M2 = self.M2 + var * count + delta ** 2 * self.count * count / new_count
        self.count = new_count
        self.M = new_M

    def reset(self):
        self.count = 0
        self.M = 0
        self.M2 = 0

    def __call__(self, x, num=1):
        self.update(x)

    @property
    def mean(self):
        # if self.count<=2:
        #     return float('nan')
        return self.M

    @property
    def var(self, samplevar=True):  # pylint: disable=property-with-parameters
        # todo: remove @property - it makes no sense here
        # if self.count<=2:
        #     return float('nan')
        return self.M2 / (self.count if samplevar else self.count - 1)

    @property
    def std(self, samplevar=True):  # pylint: disable=property-with-parameters
        # todo: remove @property - it makes no sense here
        del samplevar  # unused
        if isinstance(self.M, np.ndarray):
            return np.sqrt(self.var)

        return self.var.sqrt()

    def __repr__(self):
        return "<Welford: {} +- {}>".format(self.mean, self.std)


def register(mf):
    mf.register_event('Welford', Welford, unique=True)
