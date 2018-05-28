CXXFLAGS =	-O2 -g -Wall -fmessage-length=0

OBJS =		SBS-kaldi-2015.o

LIBS =

TARGET =	SBS-kaldi-2015

$(TARGET):	$(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LIBS)

all:	$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
