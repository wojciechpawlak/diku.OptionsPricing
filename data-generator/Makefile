CXX        = g++ -std=c++17 -O3  #-g 
CXX_GTX780 = g++ -std=c++11 -O3 -DUSE_GETOPT_ARGS #-g

SOURCES_CPP =RandomDataGenerator.cpp
EXECUTABLE  =../build/RandomDataGenerator

default: compile

.o: $(SOURCES_CPP)
	@$(CXX) -c $@ $<

compile: $(EXECUTABLE)
$(EXECUTABLE): 
	@$(CXX) -o $(EXECUTABLE) $(SOURCES_CPP)

compile_gtx780: $(EXECUTABLE)
$(EXECUTABLE): 
	@$(CXX_GTX780) -o $(EXECUTABLE) $(SOURCES_CPP)

run: $(EXECUTABLE)
	@./$(EXECUTABLE) -t $(TYPE) -n $(NUM_OPTIONS) -s $(SKEW_PERCENT)

clean:
	@rm -f $(EXECUTABLE)


