from .sum import copy
from .edge import Edge
from .variable import Variable
from .constant import Constant
from .vstack import split
from proximal.utils.timings_log import TimingsLog
import copy as cp
from collections import defaultdict
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs


class CompGraph(object):
    """A computation graph representing a composite lin op.
    """

    def __init__(self, end, implem=None):
        self.orig_end = end
        self.end = cp.copy(end)
        self.shape = self.end.shape
        # Construct via graph traversal.
        self.nodes = []
        self.edges = []
        self.constants = []
        self.input_edges = {}
        self.output_edges = {}
        new_vars = []
        # Assumes all nodes have at most one output.
        ready = [self.end]
        while len(ready) > 0:
            curr = ready.pop(0)
            if isinstance(curr, Variable):
                new_vars.append(curr)
            # Zero out constants.
            self.nodes.append(curr)
            input_edges = []
            for node in curr.input_nodes:
                # Zero out constants.
                if isinstance(node, Constant):
                    node = Constant(np.zeros(curr.shape))
                    self.constants.append(node)
                else:
                    node = cp.copy(node)
                    # Default implementation.
                    if implem is not None:
                        node.implem = implem
                ready.append(node)
                edge = Edge(node, curr, node.shape)
                input_edges.append(edge)
                self.output_edges[node] = [edge]

            self.edges += input_edges
            self.input_edges[curr] = input_edges

        # Make copy node for each variable.
        old_vars = self.orig_end.variables()
        id2copy = {}
        for var in old_vars:
            copy_node = copy(var.shape, implem=implem)
            id2copy[var.uuid] = copy_node
            self.output_edges[copy_node] = []
            self.nodes.append(copy_node)

        # Replace variables with copy nodes in graph.
        for var in new_vars:
            copy_node = id2copy[var.uuid]
            output_edge = self.output_edges[var][0]
            output_node = output_edge.end
            edge = Edge(copy_node, output_node, var.shape)
            self.edges.append(edge)
            self.output_edges[copy_node].append(edge)
            idx = self.input_edges[output_node].index(output_edge)
            self.input_edges[output_node][idx] = edge

        # Record information about variables.
        self.input_size = sum([var.size for var in old_vars])
        self.output_size = self.end.size
        self.var_info = {}

        # Make single split node as start node.
        copy_nodes = []
        offset = 0
        for key, val in id2copy.items():
            copy_nodes.append(val)
            self.var_info[key] = offset
            offset += val.size

        self.start = split(copy_nodes, implem=implem)
        self.nodes.append(self.start)
        split_outputs = []
        for copy_node in copy_nodes:
            edge = Edge(self.start, copy_node, copy_node.shape)
            split_outputs.append(edge)
            self.input_edges[copy_node] = [edge]

        self.edges += split_outputs
        self.output_edges[self.start] = split_outputs
        # A record of timings.
        self.forward_log = TimingsLog(self.nodes + [self])
        self.adjoint_log = TimingsLog(self.nodes + [self])

    def get_inputs(self, node):
        """Returns the input data for a node.
        """
        return [e.data for e in self.input_edges[node]]

    def get_outputs(self, node):
        """Returns the output data for a node.
        """
        return [e.data for e in self.output_edges[node]]

    def forward(self, x, y):
        """Evaluates the forward composition.

        Reads from x and writes to y.
        """
        def forward_eval(node):
            if node is self.start:
                inputs = [x]
            else:
                inputs = self.get_inputs(node)

            if node is self.end:
                outputs = [y]
            else:
                outputs = self.get_outputs(node)
            # Run forward op and time it.
            self.forward_log[node].tic()
            node.forward(inputs, outputs)
            self.forward_log[node].toc()
        # Evaluate forward graph and time it.
        self.forward_log[self].tic()
        self.traverse_graph(forward_eval, True)
        self.forward_log[self].toc()

    def adjoint(self, u, v):
        """Evaluates the adjoint composition.

        Reads from u and writes to v.
        """
        def adjoint_eval(node):
            if node is self.end:
                outputs = [u]
            else:
                outputs = self.get_outputs(node)

            if node is self.start:
                inputs = [v]
            else:
                inputs = self.get_inputs(node)
            # Run adjoint op and time it.
            self.adjoint_log[node].tic()
            node.adjoint(outputs, inputs)
            self.adjoint_log[node].toc()
        # Evaluate adjoint graph and time it.
        self.adjoint_log[self].tic()
        self.traverse_graph(adjoint_eval, False)
        self.adjoint_log[self].toc()

    def traverse_graph(self, node_fn, forward):
        """Traverse the graph and apply the given function at each node.

           forward: Traverse in standard or reverse order?
           node_fn: Function to evaluate on each node.
        """
        ready = []
        eval_map = defaultdict(int)
        if forward:
            ready.append(self.start)
            # Constant nodes are leaves as well.
            ready += self.constants
        else:
            ready.append(self.end)
        while len(ready) > 0:
            curr = ready.pop()
            # Evaluate the given function on curr.
            node_fn(curr)
            eval_map[curr] += 1
            if forward:
                child_edges = self.output_edges.get(curr, [])
            else:
                child_edges = self.input_edges.get(curr, [])

            # If each input has visited the node, it is ready.
            for edge in child_edges:
                if forward:
                    node = edge.end
                else:
                    node = edge.start
                eval_map[node] += 1
                if forward:
                    node_inputs_count = len(self.input_edges[node])
                else:
                    node_inputs_count = len(self.output_edges[node])

                if (eval_map[node] == node_inputs_count):
                    ready.append(node)

    def norm_bound(self, final_output_mags):
        """Returns fast upper bound on ||K||.

        Parameters
        ----------
        final_output_mags : list
            Place to store final output magnitudes.
        """
        def node_norm_bound(node):
            # Read input magnitudes off edges.
            if node is self.start:
                input_mags = [1]
            else:
                input_mags = [e.mag for e in self.input_edges[node]]

            # If a node doesn't support norm_bound, propagate that.
            if NotImplemented in input_mags:
                output_mag = NotImplemented
            else:
                output_mag = node.norm_bound(input_mags)

            if node is self.end:
                final_output_mags[0] = output_mag
            else:
                for idx, e in enumerate(self.output_edges[node]):
                    e.mag = output_mag

        self.traverse_graph(node_norm_bound, True)

    def update_vars(self, val):
        """Map sections of val to variables.
        """
        for var in self.orig_end.variables():
            offset = self.var_info[var.uuid]
            var.value = np.reshape(val[offset:offset + var.size], var.shape)
            offset += var.size

    def __str__(self):
        return self.__class__.__name__


def est_CompGraph_norm(K, tol=1e-3, try_fast_norm=True):
    """Estimates operator norm for L = ||K||.

    Parameters
    ----------
    tol : float
        Accuracy of estimate if not trying for upper bound.
    try_fast_norm : bool
        Whether to try for a fast upper bound.

    Returns
    -------
    float
        Estimate of ||K||.
    """
    if try_fast_norm:
        output_mags = [NotImplemented]
        K.norm_bound(output_mags)
        if NotImplemented not in output_mags:
            return output_mags[0]

    input_data = np.zeros(K.input_size)
    output_data = np.zeros(K.output_size)

    def KtK(x):
        K.forward(x, output_data)
        K.adjoint(output_data, input_data)
        return input_data

    # Define linear operator
    A = LinearOperator((K.input_size, K.input_size),
                       KtK, KtK)

    Knorm = np.sqrt(eigs(A, k=1, M=None, sigma=None, which='LM', tol=tol)[0].real)
    return np.float(Knorm)
