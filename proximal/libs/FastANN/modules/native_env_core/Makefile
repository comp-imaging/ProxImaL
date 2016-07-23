include ../../Makefile.inc

CFLAGS	+= -I.
OBJLIBS	= ../../libnative_env_core.a
OBJS	= ConcurrentWorkQueue.o MemAlloc.o SystemCore.o SystemTimer.o ip/ImageYUV420p.o math/BaseMath.o math/FastMath.o

all : $(OBJLIBS)

../../libnative_env_core.a : $(OBJS)
	$(AR) rv ../../libnative_env_core.a $?
	$(RANLIB) ../../libnative_env_core.a
	
-include $(OBJS:.o=.d)

%.o: %.cpp
	$(CC) $(CFLAGS) -c -o $@ $<
	$(CC) -MM $(CFLAGS) $< > $*.d
	
clean :
	-$(RM) -f *.o *.d math/*.o math/*.d ip/*.o ip/*.d $(OBJLIBS)
