###########################################################################
# compile fastann lib
###########################################################################

include ../Makefile.inc

OBJLIBS	= ../libnative_env_core.a ../libnative_env.a ../libgpu.a 
LIBFLAGS    = -L$(CUDALIBDIR) -L.. -lgpu -lnative_env -lnative_env_core -lpthread -lrt -lcudart
ADDINCLAGS  = -I$(CUDAINCDIR) -I.. -I../modules/native_env_core -I../modules/native_env -I../modules/gpu -I../src/
CXXFLAGS= $(DEFS) $(CFLAGS)
AR = ar
CXX = g++

all: $(OBJLIBS) test_denoise
default: $(OBJLIBS) test_denoise
.PHONY: default 

../libnative_env_core.a: 
	cd ../modules/native_env_core; make
	
../libnative_env.a: 
	cd ../modules/native_env; make

../libgpu.a: 
	cd ../modules/gpu; make

objs/gpuinterface.o: gpuinterface.cu
	$(NVCC) $(NVCCFLAGS) $(ADDINCLAGS) $(EXTRA_CFLAGS) -c -o "$@" "$<"

objs/%.o: %.cpp
	@echo "Building object $@"
	$(CC) $(CXXFLAGS) $(ADDINCLAGS) $< -c -o $@ 

../libdenoise.so: objs/gpuinterface.o objs/FANN_denoise.o $(OBJLIBS)
	${CC} $(CXXFLAGS) -shared -o $@ objs/gpuinterface.o objs/FANN_denoise.o $(OBJLIBS) 

#Test
test_denoise: test_denoise.cpp ../libdenoise.so
	$(CXX) $(CXXFLAGS) $(INC) test_denoise.cpp -o $@  -L.. -ldenoise -L$(CUDALIBDIR) -lcudart

cleanall:
	rm -f objs/*
	cd ../modules/native_env_core; make clean
	cd ../modules/native_env; make clean
	cd ../modules/gpu; make clean

clean:
	rm -f objs/* rm -f rm -rf ../*.a rm -rf ../*.so rm -rf test_denoise



