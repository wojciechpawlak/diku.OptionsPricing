SHELL = sh -xv
CXX        	= g++ -std=c++14 -O3 -fopenmp -I /usr/local/cuda-10.0/targets/x86_64-linux/include/ #-g
CXX_GTX780 	= g++ -std=c++11 -O3 -fopenmp -I /usr/local/cuda-9.2/targets/x86_64-linux/include/ -DUSE_GETOPT_ARGS #-g

SOURCES_CPP = SeqMain.cpp
EXECUTABLE  = ../build/Seq
DATA		= ../data

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

compile_gtx780: $(EXECUTABLE)
$(EXECUTABLE): 
	@$(CXX_GTX780) -o $(EXECUTABLE) $(DEFINE_DOUBLE) $(SOURCES_CPP)

run: $(EXECUTABLE)
	@./$(EXECUTABLE) -o $(DATA)/$(FILE).in -y $(DATA)/$(YIELD).in

test: $(EXECUTABLE)
	@./$(EXECUTABLE) -o $(DATA)/$(FILE).in -y $(DATA)/$(YIELD).in -test

repeat: $(EXECUTABLE)
	@./$(EXECUTABLE) -o $(DATA)/$(FILE).in -y $(DATA)/$(YIELD).in -test -r $(REP)

clean:
	@rm -f $(EXECUTABLE)


