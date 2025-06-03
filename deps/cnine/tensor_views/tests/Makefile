ROOTDIR=../../..
include $(ROOTDIR)/common.txt

INCLUDE= $(CNINE_INCLUDES)
INCLUDE+= -I$(TENSORVIEWDIR)/ops  

#INCLUDE= -I$(INCLUDEDIR) -I$(SCALARDIR) -I$(TENSORDIR) -I$(TENSORDIR)/ops 
#INCLUDE+= -I$(BACKENDADIR) -I$(BACKENDBDIR) -I$(TENSORVIEWDIR) -I$(TENSORVIEWDIR)/ops  
#INCLUDE+= -I$(TENSORARRAYDIR)/cell_maps 

TESTS=$(patsubst %.cpp,%,$(wildcard *.cpp))

DEPS=$(TENSORDIR)/*.hpp $(INCLUDEDIR)/*.hpp $(SCALARDIR)/*.hpp  $(TENSORVIEWDIR)/*.hpp 

EXECS= 
CUDA_OBJECTS=  
CUDA_EXTERNS=

ifdef WITH_CUDA
INCLUDE+=-I$(CUDA_HOME)/include
CUDA_OBJECTS+=$(CUDADIR)/TensorView_accumulators.o 
CUDA_OBJECTS+=$(CUDADIR)/BasicCtensorProducts.o 
CUDA_OBJECTS+=$(CUDADIR)/RtensorUtils.o  
CUDA_EXTERNS+=$(INCLUDEDIR)/Cnine_base.cu 
endif 

$(TESTS): %: %.cpp
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o $@ $@.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o $@ $@.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


objects: $(OBJECTS) $(CUDA_OBJECTS)
tests: $(TESTS)
execs: $(EXECS)
derived: $(DERIVED) 

all: objects execs derived tests 

clean: 
	rm -f $(TESTS) $(EXECS) $(OBJECTS) $(CUDA_OBJECTS) 

anew: clean all



