import ctypes as ct
import gc
import numpy as np

# 0: nothing
# 1: stdout
# 2: replay_session
verbose = 0

if verbose == 2:
    var_count = 0
    replay_session = open("/tmp/replay_ctmatlab.m", "w")

mwSize = ct.c_uint64
mxClassID = ct.c_int
size_t = ct.c_uint64

mxClassID_Values = ("mxUNKNOWN_CLASS", "mxCELL_CLASS", "mxSTRUCT_CLASS", "mxLOGICAL_CLASS", "mxCHAR_CLASS", "mxVOID_CLASS", "mxDOUBLE_CLASS", "mxSINGLE_CLASS",
                    "mxINT8_CLASS", "mxUINT8_CLASS", "mxINT16_CLASS", "mxUINT16_CLASS", "mxINT32_CLASS", "mxUINT32_CLASS", "mxINT64_CLASS", "mxUINT64_CLASS", 
                    "mxFUNCTION_CLASS")

dtype2classID = {
    'float32': mxClassID_Values.index('mxSINGLE_CLASS'),
    'float64': mxClassID_Values.index('mxDOUBLE_CLASS'),
    'int64': mxClassID_Values.index('mxINT64_CLASS'),
    'int32': mxClassID_Values.index('mxINT32_CLASS'),
    'int16': mxClassID_Values.index('mxINT16_CLASS'),
    'int8': mxClassID_Values.index('mxINT8_CLASS'),
    'uint64': mxClassID_Values.index('mxUINT64_CLASS'),
    'uint32': mxClassID_Values.index('mxUINT32_CLASS'),
    'uint16': mxClassID_Values.index('mxUINT16_CLASS'),
    'uint8': mxClassID_Values.index('mxUINT8_CLASS'),
    str: mxClassID_Values.index('mxCHAR_CLASS'),
}

classID2dtype = dict([(a,b) for b,a in dtype2classID.items()])

def CALL(v, e = ''):
    if v != 0:
        raise RuntimeError("return value != 0: %d; %s" % (v,e) )
    return v

def CALLP(v, e = ''):
    if not bool(v):
        raise RuntimeError("return value is NULL; %s" % e)
    return v

class Matlab:
    def __init__(self, pref = "/usr/local/MATLAB/R2016b/bin/glnxa64", command = None):
        self.engine = ct.CDLL(pref + "/libeng.so")
        self.engine.engOpenSingleUse.argtypes = [ct.c_char_p, ct.c_void_p, ct.POINTER(ct.c_int)]
        self.engine.engOpenSingleUse.restype = ct.c_void_p
        self.engine.engOpen.argtypes = [ct.c_char_p]
        self.engine.engOpen.restype = ct.c_void_p
        self.engine.engClose.argtypes = [ct.c_void_p]        
        self.engine.engClose.restype = ct.c_int
        self.engine.engPutVariable.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_void_p]
        self.engine.engPutVariable.restype = ct.c_int
        self.engine.engGetVariable.argtypes = [ct.c_void_p, ct.c_char_p]
        self.engine.engGetVariable.restype = ct.c_void_p
        self.engine.engEvalString.argtypes = [ct.c_void_p, ct.c_char_p]
        self.engine.engEvalString.restype = ct.c_int
        self.engine.engOutputBuffer.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_int]
        self.engine.engOutputBuffer.restype = ct.c_int
        
        self.mx = ct.CDLL(pref + "/libmx.so")
        self.mx.mxCreateNumericArray.argtypes = [mwSize, ct.POINTER(mwSize), mxClassID, ct.c_int]
        self.mx.mxCreateNumericArray.restype = ct.c_void_p
        self.mx.mxDestroyArray.argtypes = [ct.c_void_p]
        self.mx.mxGetData.argtypes = [ct.c_void_p]
        self.mx.mxGetData.restype = ct.c_void_p
        self.mx.mxGetNumberOfDimensions.argtypes = [ct.c_void_p]
        self.mx.mxGetNumberOfDimensions.restype = mwSize
        self.mx.mxGetDimensions.argtypes = [ct.c_void_p]
        self.mx.mxGetDimensions.restype = ct.POINTER(mwSize)
        self.mx.mxGetClassID.argtypes = [ct.c_void_p]
        self.mx.mxGetClassID.restype = ct.c_int
        self.mx.mxGetElementSize.argtypes = [ct.c_void_p]
        self.mx.mxGetElementSize.restype = size_t
        
        self.buf_from_mem = ct.pythonapi.PyMemoryView_FromMemory
        self.buf_from_mem.restype = ct.py_object
        self.buf_from_mem.argtypes = (ct.c_void_p, ct.c_int, ct.c_int)
        
        if command is None:
            command = ct.c_char_p()
        else:
            command = ct.c_char_p(command.encode('ascii'))
            
        ret = ct.c_int(0)
        self.eng = CALLP(self.engine.engOpen(command, ct.c_void_p(), ct.pointer(ret)))
        self.output_buffer = (ct.c_char*(1024*1024))()
        CALL(self.engine.engOutputBuffer(self.eng, self.output_buffer, ct.sizeof(self.output_buffer)))
        
    def putvar(self, name, A, nolog = False):
        if type(A) == str:
            raise NotImplemented()
        else:
            A = np.array(A)
            while len(A.shape) < 2:
                A.shape = A.shape + (1,)

            if not nolog and verbose == 1: print("(matlab)<", name, A.shape, A.dtype)

            classid = dtype2classID[A.dtype.name]
            
            dims = (mwSize*(len(A.shape)))()
            for i,d in enumerate(A.shape):
                dims[i] = d

            mxA = CALLP(self.mx.mxCreateNumericArray(len(A.shape), dims, classid, 0), name)
            try:
                d = self.mx.mxGetData(mxA)
                A = np.transpose(np.array(A)).flatten()
                ct.memmove(d, bytes(A), A.nbytes)
                CALL(self.engine.engPutVariable(self.eng, ct.c_char_p(name.encode('ascii')), mxA), name)
            finally:
                self.mx.mxDestroyArray(mxA)
            if not nolog and verbose == 2:
                global var_count
                vn = "/tmp/var%d.mat" % var_count
                var_count += 1
                self.run("save('" + vn + "', '" + name + "');", nolog=True)
                replay_session.write("tmp_ctmatlab = load('" + vn + "'); %(name)s = tmp_ctmatlab.%(name)s;\n" % locals()); 
            #int engPutVariable(Engine *ep, const char *name, const mxArray *pm);
        
    def getvar(self, name, nolog = False):
        mxA = CALLP(self.engine.engGetVariable(self.eng, ct.c_char_p(name.encode('ascii'))), name)
        try:
            ndims = self.mx.mxGetNumberOfDimensions(mxA)
            dims = self.mx.mxGetDimensions(mxA)
            classid = self.mx.mxGetClassID(mxA)
            d = self.mx.mxGetData(mxA)
            es = self.mx.mxGetElementSize(mxA)
            
            npt = classID2dtype[classid]
            shape = [0]*ndims
            for i in range(ndims):
                #print(dims[i])
                shape[i] = dims[ndims-i-1]
            
            numbytes = es * (np.prod(shape))
            #print(ndims, numbytes, shape)
            buffer = self.buf_from_mem(d, numbytes, 0x100) # read only
            res = np.transpose(np.reshape(np.frombuffer(buffer, dtype=npt), shape)).copy()
            if not nolog and verbose == 1: print("(matlab)>", name, res.shape, res.dtype)
            return res
        finally:
            self.mx.mxDestroyArray(mxA)
        
    def run(self, mcode, nargout = None, nolog = False):
        if not nolog and verbose == 1: print("(matlab)!", mcode)
        if not nolog and verbose == 2: replay_session.write(mcode + ";\n")
        mcode = """\n
try
    ct_matlab_error = 0;
    %(mcode)s
catch ME
    display(getReport(ME));
    ct_matlab_error = 1;
end
""" % locals()
        #print(mcode)
        CALL(self.engine.engEvalString(self.eng, ct.c_char_p(mcode.encode('ascii'))))
        print(self.output_buffer.value.decode('ascii'), end='')
        e = self.getvar('ct_matlab_error', nolog = True)
        if e:
            raise RuntimeError("Error while executing matlab code: " + self.output_buffer.value.decode('ascii'))
        
    def __del__(self):
        CALL(self.engine.engClose(self.eng))
        if verbose == 2: replay_session.flush()
        print("destructed")

if __name__ == "__main__":
    m = Matlab()
    A = np.reshape(np.arange(20), (4,5)).astype(np.float32)
    m.putvar('testA', A)
    m.run('testA = testA + 1;')
    m.run("display('Hello');")
    m.run("errorstatement")
    At = m.getvar('testA')
    print(A)
    print(At)
    img = np.reshape(np.arange(256*256), (256,256)).astype(np.float32)
    import timeit
    print("profile put: %.3f ms" % (timeit.timeit(lambda: m.putvar('img', img), number=1000)))
    print("profile get: %.3f ms" % (timeit.timeit(lambda: m.getvar('img'), number=1000)))
    
    m.run("this is an error")
    
    del m
    gc.collect()
    