import numpy as np

class Edge(object):
    """The edge between two lin ops.
    """
    
    edge_id = 0

    def __init__(self, start, end, shape):
        self.start = start
        self.end = end
        self.shape = shape
        self.data = np.zeros(self.shape)
        self.name = "edge%d" % Edge.edge_id
        Edge.edge_id += 1
        self.mag = None  # Used to get norm bounds.

    @property
    def size(self):
        return np.prod(self.shape)
