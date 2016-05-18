# ROOT_DIR:=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

SRC_DIR := $(CURDIR)/src
INC_DIR := $(SRC_DIR)/include
MAIN := $(SRC_DIR)/Main.cpp
HOST := $(SRC_DIR)/Host.cpp
XOR :=$(SRC_DIR)/Xor.cpp
WORKER := $(SRC_DIR)/Worker.cpp
HEADERS := $(wildcard $(INC_DIR)/*.hpp)
# SRC := $(wildcard $(SRC_DIR)/*.cpp)
# OBJ := $(SRC:.cpp=.o)

CXX := clang++
CXXFLAGS :=  -I $(INC_DIR)  -O0 -std=c++14 -stdlib=libc++ -g -Wall  -lboost_system -lpthread
# CXXFLAGS :=  -I $(INC_DIR) -O0 -std=c++14  -g -Wall 
#-Weverything -Wno-c++98-compat -Wno-padded -Wno-old-style-cast -Wno-conversion -Wno-weak-vtables -Wno-deprecated


all: local xor worker host

local: $(MAIN) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $(MAIN)

xor: $(XOR) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $(XOR)

host: $(HOST) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $(HOST)

worker: $(WORKER) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $(WORKER)

clean:
	rm -f src/*.o local xor host worker
	rm -rf *.dSYM/

install:
	cp local xor host worker ./publish
