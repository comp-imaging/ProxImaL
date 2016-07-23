include ../../Makefile.inc

CFLAGS	+= -I. -I../native_env_core
OBJLIBS	= ../../libnative_env.a
OBJS	= System.o SystemFile.o DeviceProxy.o ip/Image.o ip/RadianceReader.o

all : $(OBJLIBS)

../../libnative_env.a : $(OBJS)
	$(AR) rv ../../libnative_env.a $?
	$(RANLIB) ../../libnative_env.a
	
-include $(OBJS:.o=.d)

%.o: %.cpp
	$(CC) $(CFLAGS) -c -o $@ $<
	$(CC) -MM $(CFLAGS) $< > $*.d
	
clean :
	-$(RM) -f *.o *.d ip/*.o ip/*.d $(OBJLIBS)
