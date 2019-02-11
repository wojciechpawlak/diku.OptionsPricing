SHELL = sh -xv
CXX        	= g++ -std=c++14 -O3 -fopenmp -I /usr/local/cuda-10.0/targets/x86_64-linux/include/ #-g

SOURCES_CPP = SeqMain.cpp
EXECUTABLE  = ../build/Seq
DATA		= ../data
COMPARE		= compare

ifeq ($(REAL),64)
  DEFINE_DOUBLE = -DUSE_DOUBLE
else
  DEFINE_DOUBLE =
endif

default: run

.o: $(SOURCES_CPP)
	@$(CXX) -c $@ $<

compile: $(EXECUTABLE)
$(EXECUTABLE): 
	@$(CXX) -o $(EXECUTABLE) $(DEFINE_DOUBLE) $(SOURCES_CPP)

run: $(EXECUTABLE)
	@./$(EXECUTABLE) -o $(DATA)/$(FILE).in -y $(DATA)/$(YIELD).in

test: $(EXECUTABLE)
	@./$(EXECUTABLE) -o $(DATA)/$(FILE).in -y $(DATA)/$(YIELD).in -test

repeat: $(EXECUTABLE)
	@./$(EXECUTABLE) -o $(DATA)/$(FILE).in -y $(DATA)/$(YIELD).in -test -r $(REP)

clean:
	@rm -f $(EXECUTABLE)


