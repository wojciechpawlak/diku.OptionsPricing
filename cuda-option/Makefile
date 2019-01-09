CXX        	  = nvcc 
DEBUG_OPTIONS = -G -DCUDA_ERROR_CHECK

NVCC_FLAGS := $(ARCHS) --std=c++11 -m64 --expt-extended-lambda #--ptxas-options=-v
GCC_FLAGS := -Xcompiler '-O3 -Wall -Wextra -std=c++11 -fPIC -m64 -fopenmp'
ARCHS := -gencode arch=compute_60,code=sm_60 \
		-gencode arch=compute_35,code=sm_35

SOURCES_CPP = CudaOptionMain.cu
EXECUTABLE  = ../build/CudaOption
DATA		= ../data
COMPARE		= compare

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


