CUDALIBDIR=/usr/local/cuda/lib64
CUDAINCDIR=/usr/local/cuda/include
MEX=/home/fheide/programs/MATLAB/R2015a/bin/mex
NVCC=nvcc
CC=g++
LD=g++
DEFS=-DARCH_X86 -DLINUX -DDEBUG_MODE
NVCCFLAGS=-O3 -arch sm_30 -use_fast_math -m 64 -DARCH_X86 -DDEBUG_MODE -DLINUX -Xcompiler="-mtune=corei7 -march=corei7 -fPIC"
CFLAGS=-Wno-unused-result -O3 -ffast-math -std=gnu++0x -mtune=corei7 -DARCH_X86 -march=corei7 -fPIC -DDEBUG_MODE -DLINUX -I. -Isrc
LDFLAGS=-s
AR=ar
RANLIB=ranlib
RM=rm


