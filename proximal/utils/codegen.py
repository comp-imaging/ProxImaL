import re
import numpy as np
import functools

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import gpuarray
import pycuda.tools

def indent(code, level):
    return code.replace("\n", "\n" + (" "*level))

class NodeReverseInOut(object):
    def __init__(self, n, parent):
        self.n = n
        self.parent = parent
        
    def adjoint_cuda_kernel(self, *args, **kw):
        args = (a if not a is self.parent else self.parent.o for a in args)
        return self.n.forward_cuda_kernel(*args, **kw)
        
    def forward_cuda_kernel(self, *args,**kw):
        args = (a if not a is self.parent else self.parent.o for a in args)
        return self.n.adjoint_cuda_kernel(*args, **kw)
    
    def cuda_kernel_available(self):
        return self.n.cuda_kernel_available()
    
    @property
    def size(self):
        return self.n.size

class ReverseInOut(object):
    def __init__(self, o, reverseNodes = True):
        self.o = o
        self.reverseNodes = reverseNodes
        self.in_nodes = {}
        self.out_nodes = {}
        
    def input_nodes(self, n):
        if self.reverseNodes:
            if not n in self.in_nodes:
                self.in_nodes[n] = list([NodeReverseInOut(x, self) for x in self.o.output_nodes(n)])
            return self.in_nodes[n]
        else:
            return self.o.output_nodes(n)
    
    def output_nodes(self, n):
        if self.reverseNodes:
            if not n in self.out_nodes:
                self.out_nodes[n] = list([NodeReverseInOut(x, self) for x in self.o.input_nodes(n)])
            return self.out_nodes[n]
        else:
            return self.o.input_nodes(n)

def sub2ind(idx, shape):
    if all([not isinstance(i, str) for i in idx]):
        res = 0
        for d,i in enumerate(idx):
            res += i*np.prod(shape[(d+1):])
        return res
    else:
        res = "(" + ("+".join(["(%s)*%d" % (i, np.prod(shape[(d+1):])) for d,i in enumerate(idx)])) + ")"
    return res
        
def ind2sub(ind, shape):
    if not isinstance(ind, str):
        res = []
        for d in range(len(shape)):
            res.append( (ind % int(np.prod(shape[d:])) // int(np.prod(shape[d+1:]))) )
    else:
        res = []
        for d in range(len(shape)):
            exp1 = ""
            exp2 = ""
            if d > 0:
                exp1 = " %% %d" % int(np.prod(shape[d:]))
            if d < len(shape)-1:
                exp2 = "/ %d" % int(np.prod(shape[(d+1):]))
            res.append( "((%(ind)s)%(exp1)s)%(exp2)s" % locals() )
    return res

def replace_local_floats_with_double(src):
    Rd = re.compile(r"\bfloat\b(?! *[*])")
    Rc = re.compile(r"\b(-?(0(\.\d*)?|([1-9]\d*\.?\d*)|(\.\d+))([Ee][+-]?\d+)?)f\b")
    
    return Rc.subn(r"\1", Rd.subn("double",src)[0])[0]

class ProxyNode:
    def __init__(self, n, argname):
        self.n = n
        self.argname = argname
    
    def adjoint_cuda_kernel(self, cg, num_tmp_vars, idx, parent):
        var = "var_%d" % num_tmp_vars
        num_tmp_vars += 1
        argname = self.argname
        linidx = sub2ind(idx, self.n.shape)
        code = "float %(var)s = %(argname)s[%(linidx)s];\n" % locals()
        return code, var, num_tmp_vars
    
    def forward_cuda_kernel(self, *args):
        return self.adjoint_cuda_kernel(*args)
    
    @property
    def size(self):
        return self.n.size

class CudaSubGraph:
    """
    This class splits a linear operation graph into multiple computable subgraphs.
    """
    instance_cnt = 0
    
    def __init__(self, get_input_nodes, get_output_nodes, endnode):
        assert endnode.cuda_kernel_available()
        self.subgraph_id = CudaSubGraph.instance_cnt
        CudaSubGraph.instance_cnt += 1
        self.end = endnode
        self._input_nodes = {}
        self._output_nodes = {}
        self.kernel_nodes = []
        self.nokernel_nodes = [] # where the graph must be splitted
        self.nokernel_results = {}
        self.nokernel_inputs = {}
        self.nokernel_proxynodes = {}
        active_nodes = [self.end]
        visited_nodes = {}
        nokernel_innodes = []
        while len(active_nodes) > 0:
            n = active_nodes.pop(0)
            if n in visited_nodes:
                continue
            visited_nodes[n] = True
            try:
                self._output_nodes[n] = get_output_nodes(n)
            except KeyError:
                pass
            if n.cuda_kernel_available():
                self.kernel_nodes.append(n)
                try:
                    innodes = get_input_nodes(n)
                    self._input_nodes[n] = innodes
                    active_nodes.extend(innodes)
                except KeyError:
                    pass
            else:       
                cn = [n]
                while 1:
                    innodes = get_input_nodes(cn[0])
                    assert (len(innodes) == 1)
                    assert (len(get_output_nodes(cn[0])) == 1)
                    if innodes[0].cuda_kernel_available():
                        nokernel_innodes += innodes
                        break
                    cn = [innodes[0]] + cn
                self.nokernel_nodes.append(cn)
        self.dependent_subgraphs = []
        for n in nokernel_innodes:
            dsg = CudaSubGraph(get_input_nodes, get_output_nodes, n)
            self.dependent_subgraphs.append(dsg)
        
        #print(str(self))
        
    def __str__(self):
        res = "CudaSubGraph(\n"
        for n in self.kernel_nodes:
            res += "  %s <- " % repr(n)
            res += ", ".join(["%s" % repr(x) for x in self._input_nodes.get(n, [])])
            res += "\n"
        res += ")"
        return res
    
    def input_nodes(self, n):
        innodes = self._input_nodes[n]
        res = []
        for inn in innodes:
            noKernelNode = False
            for cn in self.nokernel_nodes:
                if inn in cn:
                    noKernelNode = True
            if noKernelNode:
                res.append(self.nokernel_proxynodes[inn])
            else:
                res.append(inn)
        #print("innodes(%s) -> %s" % (repr(n), res))
        return res
    
    def output_nodes(self, n): # this is called for adjoint operation
        return self._output_nodes[n]
        
    def gen_code(self, fcn, parent = None):
        """
        generates the cuda kernel code. fcn should either be "forward_cuda_kernel" or "adjoint_cuda_kernel"
        """
        self.cuda_args = []
        self.fcn = fcn
        for n in self.kernel_nodes:
            try:
                buffers = n.cuda_additional_buffers()
            except AttributeError:
                buffers = []
            for aname, aval in buffers:
                aval = gpuarray.to_gpu(aval.astype(np.float32))
                self.cuda_args.append( (aname, aval) )
                
        for cn in self.nokernel_nodes:
            for n in cn:
                o = gpuarray.zeros(n.shape, dtype=np.float32)
                self.nokernel_results[n] = o
            n = cn[-1]
            self.cuda_args.append( ("linop_proxy_output_%d" % n.linop_id, o) )
            self.nokernel_proxynodes[n] = ProxyNode(n, self.cuda_args[-1][0])
            
        add_args = "".join((", float *%s" % x[0] for x in self.cuda_args))
        
        cg = self if fcn == "forward_cuda_kernel" else ReverseInOut(self, reverseNodes=False)
        self.shape = parent.shape if not parent is None else self.end.shape
        cucode, var, num_tmp_vars = getattr(self.end, fcn)(cg, 0, ind2sub("yidx", self.shape), parent)
        cucode = indent(cucode, 8)
        dimy = int(np.prod(self.shape))
        subgraph_id = self.subgraph_id
        code  = """\
__global__ void %(fcn)s_%(subgraph_id)d(const float *x, float *y%(add_args)s)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x; 
    int stride = blockDim.x * gridDim.x;
    for( int yidx = index; yidx < %(dimy)d; yidx += stride )
    {
        %(cucode)s
        y[yidx] = %(var)s;
    }
}
    
""" % locals()

        for i,dsg in enumerate(self.dependent_subgraphs):
            cn = self.nokernel_nodes[i]
            dsg.gen_code(fcn, cn[0])
            lastShape = dsg.shape
            for ni, n in enumerate(cn):
                self.nokernel_inputs[n] = gpuarray.zeros(lastShape, dtype=np.float32) if ni == 0 else self.nokernel_results[cn[ni-1]]
                lastShape = n.shape
        
        try:
            self._cuda_code = code
            self.cuda_mod = SourceModule(code)
        except cuda.CompileError as e:
            print(code)
            print("CUDA compilation error:")
            print(e.stderr)
            raise e
                    
        cuda_func = self.cuda_mod.get_function("%(fcn)s_%(subgraph_id)d" % locals())
        block = (min(int(dimy), cuda_func.MAX_THREADS_PER_BLOCK), 1, 1)
        grid = (int(dimy)//block[0],1,1)
        arg_vals = tuple(x[1] for x in self.cuda_args)
        self.cuda_kernel_func = lambda *args: cuda_func(*(args+arg_vals), grid=grid, block=block, time_kernel=True)
        
    def apply(self, x, y):
        t = 0.0
        for i,dsg in enumerate(self.dependent_subgraphs):
            cn = self.nokernel_nodes[i]
            n = cn[0]
            t += dsg.apply(x, self.nokernel_inputs[n])
        for i,cn in enumerate(self.nokernel_nodes):
            if self.fcn == "forward_cuda_kernel":
                for n in cn:
                    n.forward_cuda([self.nokernel_inputs[n]], [self.nokernel_results[n]])
            else:
                for n in cn:
                    n.adjoint_cuda([self.nokernel_inputs[n]], [self.nokernel_results[n]])
        t += self.cuda_kernel_func(x, y)
        self.output = gpuarray.reshape(y, self.shape)
        return t
        
    @property
    def cuda_code(self):
        return self._cuda_code + "\n".join([x.cuda_code for x in self.dependent_subgraphs])

