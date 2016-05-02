# ROOT_DIR:=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

SRC_DIR := $(CURDIR)/src
INC_DIR := $(SRC_DIR)/include
MAIN := $(SRC_DIR)/Main.cpp
# SRC := $(filter-out $(MAIN), $(wildcard $(SRC_DIR)/*.cpp))
HEADERS := $(wildcard $(INC_DIR)/*.hpp)
# SRC := $(wildcard $(SRC_DIR)/*.cpp)
# OBJ := $(SRC:.cpp=.o)

CXX := clang++
CXXFLAGS :=  -I $(INC_DIR) -O3 -std=c++14 -stdlib=libc++ -g -Wall -Weverything -Wno-c++98-compat -Wno-padded -Wno-old-style-cast


all: test

test: $(MAIN) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $(MAIN)


clean:
	rm -f src/*.o test
	rm -rf *.dSYM/

# %.o: %.cpp
	# $(CXX) $(CXXFLAGS) -c -o $@ $<
