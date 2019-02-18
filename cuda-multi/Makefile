SHELL = sh -xv
CXX        	  = nvcc
DEBUG_OPTIONS = -G -DCUDA_ERROR_CHECK

ARCHS := -gencode arch=compute_70,code=sm_70
NVCC_FLAGS := $(ARCHS) --std=c++14 -m64 --expt-extended-lambda -Xptxas=-v
GCC_FLAGS := -Xcompiler '-O3 -Wall -Wextra -std=c++14 -fPIC -m64 -fopenmp'

ARCHS_GTX780 := -gencode arch=compute_35,code=sm_35
NVCC_FLAGS_GTX780 := $(ARCHS_GTX780) --std=c++11 -m64 --expt-extended-lambda -Xptxas=-v
GCC_FLAGS_GTX780 := -Xcompiler '-O3 -Wall -Wextra -std=c++11 -fPIC -m64 -fopenmp -DUSE_GETOPT_ARGS'

SOURCES_CPP = CudaMultiMain.cu
EXECUTABLE  = ../build/CudaMulti
DATA		= ../data

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
	@$(CXX) -c $@ $<

compile: $(EXECUTABLE)
$(EXECUTABLE): 
	@$(CXX) $(NVCC_FLAGS) $(GCC_FLAGS) $(REG_COUNT) $(DEFINE_DOUBLE) -o $(EXECUTABLE) $(SOURCES_CPP)

compile_gtx780: $(EXECUTABLE)
$(EXECUTABLE): 
	@$(CXX) $(NVCC_FLAGS_GTX780) $(GCC_FLAGS_GTX780) $(REG_COUNT) $(DEFINE_DOUBLE) -o $(EXECUTABLE) $(SOURCES_CPP)

compile-debug: $(EXECUTABLE)
	@$(CXX) $(NVCC_FLAGS) $(GCC_FLAGS) $(REG_COUNT) $(DEFINE_DOUBLE) $(DEBUG_OPTIONS) -o $(EXECUTABLE) $(SOURCES_CPP)

run: $(EXECUTABLE)
	@./$(EXECUTABLE) -o $(DATA)/$(FILE).in -y $(DATA)/$(YIELD).in -s $(SORT) -v $(VERSION) -b $(BLOCK) -d $(DEVICE)

test: $(EXECUTABLE)
	@./$(EXECUTABLE) -o $(DATA)/$(FILE).in -y $(DATA)/$(YIELD).in -s $(SORT) -v $(VERSION) -b $(BLOCK) -d $(DEVICE) -t

repeat: $(EXECUTABLE)
	@./$(EXECUTABLE) -o $(DATA)/$(FILE).in -y $(DATA)/$(YIELD).in -s $(SORT) -v $(VERSION) -b $(BLOCK) -d $(DEVICE) -t -r $(REP)

clean:
	@rm -f $(EXECUTABLE)


