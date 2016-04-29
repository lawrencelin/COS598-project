# ROOT_DIR:=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

SRC_DIR := $(CURDIR)/src
INC_DIR := $(SRC_DIR)/include
MAIN := $(SRC_DIR)/Main.cpp
# SRC := $(filter-out $(MAIN), $(wildcard $(SRC_DIR)/*.cpp))
HEADERS := $(wildcard $(INC_DIR)/*.hpp)
# SRC := $(wildcard $(SRC_DIR)/*.cpp)
# OBJ := $(SRC:.cpp=.o)

CXX := clang++
CXXFLAGS :=  -I $(INC_DIR) -O0 -std=c++14 -stdlib=libc++ -g -Wall
# -Weverything -Wno-c++98-compat -Wno-padded -Wno-header-hygiene -Wno-sign-compare -Wno-shorten-64-to-32 -Wno-unused-parameter -Wno-missing-field-initializers  -Wno-sign-conversion -Wno-shadow -Wno-exit-time-destructors -Wno-global-constructors -Wno-return-type

all: test

# test: $(MAIN) $(OBJ)
test: $(MAIN) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $(MAIN)


clean:
	rm -f src/*.o test
	rm -rf *.dSYM/

# %.o: %.cpp
	# $(CXX) $(CXXFLAGS) -c -o $@ $<
