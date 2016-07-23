###########################################################################
# mex interface compilation
# user-configurable section
###########################################################################

include ../Makefile.inc

OBJLIBS	= ../libnative_env_core.a ../libnative_env.a ../libgpu.a
LIBFLAGS    = -L$(CUDALIBDIR) -L.. -lgpu -lnative_env -lnative_env_core -lpthread -lrt -lcudart
ADDINCLAGS  = -I$(CUDAINCDIR) -I.. -I../modules/native_env_core -I../modules/native_env -I../modules/gpu -I../src/

##########################################################################

# mex flags
MEXFLAGS=-O $(INCLUDE) $(DEFS) CXXFLAGS="-mtune=corei7 -march=corei7 -fPIC"

###########################################################################


all: $(OBJLIBS) mexFastDenoiseOptWrapper mexFastIndexWrapper mexFastQueryWrapper mexGtQueryWrapper
../libnative_env_core.a: 
	cd ../modules/native_env_core; make
	
../libnative_env.a: 
	cd ../modules/native_env; make

../libgpu.a: 
	cd ../modules/gpu; make


objs/gpuinterface.o: gpuinterface.cu
	$(NVCC) $(NVCCFLAGS) $(ADDINCLAGS) $(EXTRA_CFLAGS) -c -o "$@" "$<"

objs/gpuinterface_separate.o: gpuinterface_separate.cu
	$(NVCC) $(NVCCFLAGS) $(ADDINCLAGS) $(EXTRA_CFLAGS) -c -o "$@" "$<"

default: mexFastDenoiseOptWrapper

.PHONY: default 

objs/%.o: %.cpp
	@echo "Building object $@"
	@$(MEX) $(MEXFLAGS) $(ADDINCLAGS) -outdir ./objs -c $<

mexFastDenoiseOptWrapper: objs/fastdenoise_mex.o $(OBJLIBS) objs/gpuinterface.o
	@echo "Linking $@"
	@$(MEX) -cxx $(MEXFLAGS) $(LIBFLAGS) objs/fastdenoise_mex.o $(OBJLIBS) objs/gpuinterface.o

mexFastIndexWrapper: objs/fastbuildindex_mex.o $(OBJLIBS) objs/gpuinterface_separate.o
	@echo "Linking $@"
	@$(MEX) -cxx $(MEXFLAGS) $(LIBFLAGS) objs/fastbuildindex_mex.o $(OBJLIBS) objs/gpuinterface_separate.o 

mexFastQueryWrapper: objs/fastquery_mex.o $(OBJLIBS) objs/gpuinterface_separate.o
	@echo "Linking $@"
	@$(MEX) -cxx $(MEXFLAGS) $(LIBFLAGS) objs/fastquery_mex.o $(OBJLIBS) objs/gpuinterface_separate.o

mexGtQueryWrapper: objs/query_gt_mex.o $(OBJLIBS)
	@echo "Linking $@"
	@$(MEX) -cxx $(MEXFLAGS) $(LIBFLAGS) objs/query_gt_mex.o $(OBJLIBS)
	
cleanall:
	rm -f objs/* rm -f *.mexglx *.mexa64
	cd ../modules/native_env_core; make clean
	cd ../modules/native_env; make clean
	cd ../modules/gpu; make clean

clean:
	rm -f objs/* rm -f *.mexglx *.mexa64



