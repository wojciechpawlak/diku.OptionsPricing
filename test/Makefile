CXX	= nvcc
CPP	= g++ 

ARCHS := -gencode arch=compute_70,code=sm_70
NVCC_FLAGS := $(ARCHS) --std=c++11 -m64 --expt-extended-lambda -Xcompiler '-O3 -Wall -Wextra -std=c++11 -fPIC -m64 -fopenmp'
GCC_FLAGS := -O3 -Wall -Wextra -std=c++11 -fPIC -m64 -fopenmp

ARCHS_GTX780 := -gencode arch=compute_35,code=sm_35
NVCC_FLAGS_GTX780 := $(ARCHS_GTX780) --std=c++11 -m64 --expt-extended-lambda -Xptxas=-v
GCC_FLAGS_GTX780 := -Xcompiler '-O3 -Wall -Wextra -std=c++11 -fPIC -m64 -fopenmp -DUSE_GETOPT_ARGS'

SOURCES_CPP = TestMain.cpp TestCuda.cu
EXECUTABLE  = ../build/Test

SOURCES_COMPARE		= Compare.cpp
EXECUTABLE_COMPARE  = ../build/Compare

ifeq ($(REAL),64)
  DEFINE_DOUBLE = -DUSE_DOUBLE
else
  DEFINE_DOUBLE =
endif

ifeq ($(REG),32)
  REG_COUNT = --maxrregcount=32
else
  REG_COUNT =
endif

default: run

.cu.o: $(SOURCES_CPP)
	@$(CXX) $(NVCC_FLAGS)  -c $@ $<

compile: $(EXECUTABLE)
$(EXECUTABLE): 
	@$(CXX) $(NVCC_FLAGS) $(REG_COUNT) $(DEFINE_DOUBLE) -o $(EXECUTABLE) $(SOURCES_CPP)

compile_gtx780: $(EXECUTABLE)
$(EXECUTABLE): 
	@$(CXX) $(NVCC_FLAGS_GTX780) $(GCC_FLAGS_GTX780) $(REG_COUNT) $(DEFINE_DOUBLE) -o $(EXECUTABLE) $(SOURCES_CPP)

compile-compare: $(EXECUTABLE_COMPARE)
$(EXECUTABLE_COMPARE):
	@$(CPP) $(GCC_FLAGS) $(DEFINE_DOUBLE) -o $(EXECUTABLE_COMPARE) $(SOURCES_COMPARE)

compare: $(EXECUTABLE_COMPARE)
	@cat $(DATA)/$(FILE1).out $(FILE2).out | ./$(EXECUTABLE_COMPARE)

run: $(EXECUTABLE)
	@./$(EXECUTABLE)

clean:
	@rm -f $(EXECUTABLE) $(EXECUTABLE_COMPARE)
