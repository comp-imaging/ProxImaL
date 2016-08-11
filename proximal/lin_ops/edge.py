import numpy as np


class Edge(object):
    """The edge between two lin ops.
    """

    def __init__(self, start, end, shape):
        self.start = start
        self.end = end
        self.shape = shape
        self.data = np.zeros(self.shape)
        self.mag = None  # Used to get norm bounds.

    @property
    def size(self):
        return np.prod(self.shape)
