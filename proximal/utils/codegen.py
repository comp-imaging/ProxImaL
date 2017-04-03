import re
import numpy as np

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
    
    @property
    def size(self):
        return self.n.size

class ReverseInOut(object):
    def __init__(self, o):
        self.o = o
        self.in_nodes = {}
        self.out_nodes = {}
        
    def input_nodes(self, n):
        if not n in self.in_nodes:
            self.in_nodes[n] = list([NodeReverseInOut(x, self) for x in self.o.output_nodes(n)])
        return self.in_nodes[n]
    
    def output_nodes(self, n):
        if not n in self.out_nodes:
            self.out_nodes[n] = list([NodeReverseInOut(x, self) for x in self.o.input_nodes(n)])
        return self.out_nodes[n]

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
