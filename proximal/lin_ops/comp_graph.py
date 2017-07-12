from .sum import copy
from .edge import Edge
from .variable import Variable
from .constant import Constant
from .vstack import split
from ..utils.cuda_codegen import CudaSubGraph, gpuarray
from .vstack import vstack
from proximal.utils.timings_log import TimingsLog
import copy as cp
from collections import defaultdict
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs

class CompGraph(object):
    """A computation graph representing a composite lin op.
    """
    
    instanceCnt = 0

    def __init__(self, end, implem=None):
        self.instanceID = CompGraph.instanceCnt
        CompGraph.instanceCnt += 1
        self.orig_end = end
        self.end = cp.copy(end)
        self.end.orig_node = end.orig_node
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
        done = []
        node_to_copies = {}
        self.split_nodes = {}
        while len(ready) > 0:
            curr = ready.pop(0)
            done.append(curr)
            if isinstance(curr, Variable):
                # new_vars may contain specific variables more than once
                new_vars.append(curr)
            # Zero out constants.
            self.nodes.append(curr)
            input_edges = []
            for node in curr.input_nodes:
                # Zero out constants. Constants are handled in absorb_offset
                if isinstance(node, Constant):
                    node = Constant(np.zeros(curr.shape))
                    node.orig_node = None
                    self.constants.append(node)
                else:
                    # avoid copying too many nodes
                    if not node in node_to_copies:                            
                        cnode = cp.copy(node)
                        node.orig_node = node.orig_node
                        node_to_copies[node] = cnode
                    else:
                        self.split_nodes[node_to_copies[node]] = True
                    node = node_to_copies[node]
                    # Default implementation.
                    if implem is not None:
                        node.implem = implem
                if not node in ready and not node in done:
                    ready.append(node)
                edge = Edge(node, curr, node.shape)
                input_edges.append(edge)
                if not node in self.output_edges:
                    self.output_edges[node] = [edge]
                else:
                    self.output_edges[node].append(edge)

            self.edges += input_edges
            self.input_edges[curr] = input_edges

        # replace the split nodes with copy nodes
        for n in self.split_nodes.keys():
            outedges = self.output_edges[n]
            outnodes = [e.end for e in outedges]
            copy_node = copy(n, implem=implem)
            copy_node.input_nodes += [n]
            self.output_edges[n] = [Edge(n, copy_node, n.shape)]
            self.input_edges[copy_node] = self.output_edges[n]
            self.output_edges[copy_node] = []
            self.nodes.append(copy_node)
            for ns in outnodes:
                inedges = self.input_edges[ns]
                newinedges = []
                for e in inedges:
                    if e.start is n:
                        e = Edge(copy_node, e.end, copy_node.shape)
                        newinedges.append( e )
                        self.output_edges[copy_node].append(e)
                    else:
                        newinedges.append( e )
                self.input_edges[ns] = newinedges            

        # Make copy node for each variable.
        old_vars = self.orig_end.variables()
        id2copy = {}
        copy_nodes = []
        self.var_info = {}
        offset = 0
        for var in old_vars:
            copy_node = copy(var.shape, implem=implem)
            copy_node.orig_node = None
            id2copy[var.uuid] = copy_node
            copy_nodes.append(copy_node)
            self.var_info[var.uuid] = offset
            offset += copy_node.size
            self.output_edges[copy_node] = []
            self.nodes.append(copy_node)

        # Replace variables with copy nodes in graph.
        for var in new_vars:
            copy_node = id2copy[var.uuid]
            for output_edge in self.output_edges[var]:
                output_node = output_edge.end
                edge = Edge(copy_node, output_node, var.shape)
                self.edges.append(edge)
                self.output_edges[copy_node].append(edge)
                idx = self.input_edges[output_node].index(output_edge)
                #print("Variable %s(%s): idx=%d" % (var.varname, var.uuid, idx))
                self.input_edges[output_node][idx] = edge

        # Record information about variables.
        self.input_size = sum([var.size for var in old_vars])
        self.output_size = self.end.size

        self.start = split(copy_nodes, implem=implem)
        self.start.orig_node = None
        self.nodes.append(self.start)
        split_outputs = []
        for copy_node in copy_nodes:
            edge = Edge(self.start, copy_node, copy_node.shape)
            split_outputs.append(edge)
            self.input_edges[copy_node] = [edge]

        self.edges += split_outputs
        self.output_edges[self.start] = split_outputs
        
        self.cuda_forward_subgraphs = None
        self.cuda_adjoint_subgraphs = None
        
        # A record of timings.
        self.forward_log = TimingsLog(self.nodes + [self])
        self.adjoint_log = TimingsLog(self.nodes + [self])
        
    def input_nodes(self, node):
        return list([e.start for e in self.input_edges[node]])

    def output_nodes(self, node):
        return list([e.end for e in self.output_edges[node]])

    def get_inputs(self, node):
        """Returns the input data for a node.
        """
        return [e.data for e in self.input_edges[node]]

    def get_outputs(self, node):
        """Returns the output data for a node.
        """
        return [e.data for e in self.output_edges[node]]

    def gen_cuda_code(self):
        # The basic original idea is to generate a cuda kernel for the whole graph
        # this is done by calling the output node (self.end for forward direction,
        # self.start for adjoint direction), and these nodes will recursively 
        # generate the kernel operations also for their input nodes.
        #
        # There are certain nodes in the graph which are either not yet ported to
        # cuda or they don't efficiently fit into the above scheme. For example,
        # it is not efficient to perform a convolution operation in the middle of
        # a long linear graph (because the preceding image values have to be 
        # calculated size(conv_kernel) times). Therefore we need a way to split
        # the graph into subgraphs, and calculate individual nodes for their own.
        #
        # Nodes who want to be isolated can either not implement the forward_cuda_kernel
        # function or override the function cuda_kernel_available(self) and return false.
        
        # forward direction
        self.cuda_forward_subgraphs = CudaSubGraph(self.input_nodes, self.output_nodes, self.end)
        self.cuda_forward_subgraphs.gen_code("forward_cuda_kernel")
        #print("Forward subgraphs:")
        #self.cuda_forward_subgraphs.visualize()
        
        self.cuda_adjoint_subgraphs = CudaSubGraph(self.output_nodes, self.input_nodes, self.start)
        self.cuda_adjoint_subgraphs.gen_code("adjoint_cuda_kernel")
        #print("Adjoint subgraphs:")
        #self.cuda_adjoint_subgraphs.visualize()
                                                                         
    def forward_cuda(self, x, y, printt=False):
        if 0:
            needcopy = False
            if type(x) is gpuarray.GPUArray:
                x = x.get()
            if type(y) is gpuarray.GPUArray:
                needcopy = True
                yorig = y
                y = y.get()
            self.forward(x, y)
            if needcopy:
                yorig[:] = gpuarray.to_gpu(y)
        else:
            if self.cuda_forward_subgraphs is None:
                self.gen_cuda_code()
            if not type(x) is gpuarray.GPUArray:
                x = gpuarray.to_gpu(x.astype(np.float32))
            if not type(y) is gpuarray.GPUArray:
                y = gpuarray.to_gpu(y.astype(np.float32))
                print("Warning: result y is no GPU array.")
            self.forward_log[self].tic()
            t = self.cuda_forward_subgraphs.apply(x, y)
            self.forward_log[self].toc()
            if printt: print(t)
        return y

    def adjoint_cuda(self, y, x, printt=False):
        if 0:
            needcopy = False
            if type(x) is gpuarray.GPUArray:
                needcopy = True
                xorig = x
                x = x.get()
            if type(y) is gpuarray.GPUArray:
                y = y.get()
            self.adjoint(y, x)
            if needcopy:
                xorig[:] = gpuarray.to_gpu(x)            
        else:
            if self.cuda_adjoint_subgraphs is None:
                self.gen_cuda_code()
            if not type(x) is gpuarray.GPUArray:
                x = gpuarray.to_gpu(x.astype(np.float32))
                print("Warning: result x is no GPU array.")
            if not type(y) is gpuarray.GPUArray:
                y = gpuarray.to_gpu(y.astype(np.float32))
            self.adjoint_log[self].tic()
            t = self.cuda_adjoint_subgraphs.apply(y, x)
            self.adjoint_log[self].toc()
            if printt: print(t)
        return x
        
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
            #if node in self.split_nodes:
            #    for io in range(1,len(outputs)):
            #        np.copyto(outputs[io], outputs[0])
            self.forward_log[node].toc()
            
        self.forward_log[self].tic()
        # Evaluate forward graph and time it.
        self.traverse_graph(forward_eval, True)
        self.forward_log[self].toc()
        return y

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
            #if node in self.split_nodes:
            #    for io in range(len(outputs)):
            #        node.adjoint([outputs[io]], inputs)
            #        assert(len(inputs) == 1)
            #        if io == 0:
            #            res = inputs[0].copy()
            #        else:
            #            res += inputs[0]
            #    np.copyto(inputs[0], res)
            #else:
            if 1:
                node.adjoint(outputs, inputs)
            self.adjoint_log[node].toc()
        # Evaluate adjoint graph and time it.
        self.adjoint_log[self].tic()
        self.traverse_graph(adjoint_eval, False)
        self.adjoint_log[self].toc()
        return v

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
            
    def x0(self):
        res = np.zeros(self.input_size)
        for var in self.orig_end.variables():
            if var.initval is not None:
                offset = self.var_info[var.uuid]
                res[offset:offset + var.size] = np.ravel(var.initval)
        return res
    
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
