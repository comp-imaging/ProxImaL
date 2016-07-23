include ../../Makefile.inc

NVCCFLAGS += -I.
OBJLIBS = ../../libgpu.a
OBJS = Core/Internal/FrameBuffer.o Core/FrameBufferHandle.o Core/FrameBufferManager.o Core/Internal/FrameBufferImpl.o

all : $(OBJLIBS)

../../libgpu.a: $(OBJS)
	$(NVCC) $(NVCCFLAGS) -lib -o "$@" $(OBJS)

Core/Internal/FrameBufferImpl.o: Core/Internal/FrameBufferImpl.cu
	$(NVCC) $(NVCCFLAGS) $(EXTRA_CFLAGS) -c -o "$@" "$<"

%.o: %.cpp
	$(NVCC) $(NVCCFLAGS) -c -o "$@" "$<"

clean:
	-$(RM) -f $(OBJS) $(OBJLIBS) 
