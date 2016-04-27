MAIN := src/Main.cpp
SRC := $(filter-out $(MAIN), $(wildcard src/*.cpp))
OBJ := $(SRC:.cpp=.o)

CXX := clang++
CXXFLAGS :=  -I src/include/ -O3 -std=c++11 -stdlib=libc++ -g -Wall
# -Weverything -Wno-c++98-compat -Wno-padded -Wno-header-hygiene -Wno-sign-compare -Wno-shorten-64-to-32 -Wno-unused-parameter -Wno-missing-field-initializers  -Wno-sign-conversion -Wno-shadow -Wno-exit-time-destructors -Wno-global-constructors -Wno-return-type

all: test

test: $(MAIN) $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^


clean:
	rm -f src/*.o test
	rm -rf *.dSYM/

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<
