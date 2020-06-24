TARGET = ./public/main

SRCS := $(wildcard ./src/*.cpp ./*.cpp)

OBJS := $(patsubst %cpp,%o,$(SRCS))

CFLG = -g -Wall -I/usr/local/Cellar/opencv/4.3.0_5/include/opencv4 -Iinc -I./ -std=c++17 -stdlib=libc++ --debug

LDFG = -Wl, $(shell pkg-config opencv4 --cflags --libs)

CXX = g++
 
$(TARGET) : $(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LDFG)

%.o:%.cpp
	$(CXX) $(CFLG) -c $< -o $@ 

.PHONY : clean
clean:
	-rm ./**/*.o
  