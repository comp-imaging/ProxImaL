import numpy as np
import gc
import tempfile
import os.path

mlengine = 'ctypes'

if mlengine == 'matlab.engine':

    import matlab.engine
    import matlab
    
    _engw = None
    class EngineWrapper:
        def __init__(self):
            self._eng = matlab.engine.connect_matlab(async=True).result()
            
        def __del__(self):
            print("disconnecting from matlab")
            self._eng.exit()
            
        def get(self):
            return self._eng
            
    t = {np.dtype('float64'): 'double',
         np.dtype('float32'): 'single',
         np.dtype('int64'): 'int64',
         np.dtype('int32'): 'int32',
         np.dtype('int16'): 'int16',
         np.dtype('int8'): 'int8',
         np.dtype('uint64'): 'uint64',
         np.dtype('uint32'): 'uint32',
         np.dtype('uint16'): 'uint16',
         np.dtype('uint8'): 'uint8',
         }
    
    def _connect():
        global _engw
        if _engw is None:
            gc.collect()
            _engw = EngineWrapper()
            
    def disconnect():
        global _engw
        if not _engw is None:
            del _engw
            gc.collect()
            _engw = None
    
    def engine():
        _connect()
        return _engw.get()
    
    def put_array(name, A, globalvar = False):
        _connect()
        _eng = _engw.get()
        A = np.array(A)
        sA = A.shape
        if len(sA) == 1:
            sA = (sA[0], 1)
        elif len(sA) == 0:
            sA = (1,1)
        tA = t[A.dtype]
        A = bytes(A.flatten(order='F'))
        _eng.workspace[name] = A
        _eng.workspace[name] = _eng.reshape(_eng.typecast(A, tA), matlab.double(sA))
        
    def get_array(name):
        _eng = _engw.get()
        a = _eng.workspace[name]
        return np.reshape(np.array(a._data, order='F'), a.size[::-1]).T

elif mlengine == 'ctypes':
    try:
        from .ctmatlab import Matlab
    except SystemError:
        from ctmatlab import Matlab
    
    _engw = None
    class EngineWrapper:
        def __init__(self):
            self._eng = Matlab()
            
        def __del__(self):
            print("disconnecting from matlab")
            del self._eng
            gc.collect()
            
        def get(self):
            return self._eng
    
    def _connect():
        global _engw
        if _engw is None:
            gc.collect()
            _engw = EngineWrapper()
            
    def disconnect():
        global _engw
        if not _engw is None:
            del _engw
            gc.collect()
            _engw = None
    
    def engine():
        _connect()
        return _engw.get()
    
    def put_array(name, A, globalvar = False):
        _connect()
        if globalvar:
            _engw.get().run("global " + name)
        _engw.get().putvar(name, A)
        
    def get_array(name):
        _connect()
        return _engw.get().getvar(name)
    
else:
    
    raise NotImplemented()


class MatlabClass:
    def __init__(self, name):
        self.stream = tempfile.NamedTemporaryFile('w+', suffix='.m', prefix=name + "_", delete = False)
        self.classname =  os.path.split(os.path.splitext(self.stream.name)[0])[1]
        self.instancename = "i_%s" % self.classname
        self.properties = ["d"]
        self.methods = []
        
    def add_method(self, m, constructor = ""):
        self.methods.append((m, constructor))
        
    def generate(self):
        def indent(s, l):
            i = " "*l
            return i + s.replace("\n", "\n" + i)
        
        classname = self.classname
        
        properties = indent("\n".join(self.properties), 8)
        methods = indent("\n\n\n".join([x[0] for x in self.methods]), 8)
        constructors = indent("\n".join([ x[1] for x in self.methods ]), 12)
        
        self.stream.write("""
classdef %(classname)s
    properties
        %(properties)s
    end
    
    methods
        function obj = %(classname)s()
            obj.d = struct();
            %(constructors)s;
        end
    
        %(methods)s
    end
end
""" % locals())
        
        self.stream.flush()
        instancename = self.instancename
        eng = engine()
        eng.run("addpath('" + os.path.split(self.stream.name)[0] + "');")
        eng.run("%(instancename)s = %(classname)s();" % locals())

if __name__ == "__main__":
    def check_equal(npa, mla):
        idx = np.zeros( len(npa.shape), dtype=np.int64 )
        fin = False
        while not fin:
            i = list(zip(idx))
            v1 = npa[i]
            v2 = mla
            for i in idx:
                v2 = v2[i]
            #print(idx, v1, v2)
            assert(v1 == v2)
            k = len(idx)-1
            idx[k] += 1
            while idx[k] >= npa.shape[k]:
                idx[k] = 0
                if k == 0:
                    fin = True
                    break
                k -= 1
                idx[k] += 1
    
    s = 5;
    put_array("test_s", s)
    
    a = np.array([1,2,3,4])
    put_array("test_a", a)
    check_equal(a, get_array("test_a"))
    
    b = np.reshape(np.array(range(20)), (4,5))
    put_array("test_b", b)
    check_equal(b, get_array("test_b"))
    
    c = np.reshape(np.array(range(2*3*4)), (2,3,4))
    put_array("test_c", c)
    check_equal(c, get_array("test_c"))
    
    bt = get_array('test_b')
    assert(np.all(b == bt))
    
    ct = get_array('test_c')
    assert(np.all(c == ct))
    