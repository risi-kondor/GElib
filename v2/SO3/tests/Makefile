ROOTDIR=../..
include $(ROOTDIR)/common.txt

INCLUDE= $(CNINE_INCLUDES) 
INCLUDE+=-I$(INCLUDEDIR) -I$(COREDIR) -I$(SO3DIR)  

TESTS=$(patsubst %.cpp,%,$(wildcard *.cpp))

#TESTS+= testSO3partB_array testSO3partB_array2 testSO3vecB_array 

ifdef WITH_CUDA
TESTS+= 
endif 

DEPS=$(SO3DIR)/*.hpp 

EXECS= 

CUDA_OBJECTS=  
CUDA_EXTERNS=

ifdef WITH_CUDA
INCLUDE+=-I$(CUDA_HOME)/include
CUDA_OBJECTS+=$(CNINE_CUDADIR)/RtensorUtils.o $(CNINE_CUDADIR)/TensorView_accumulators.o  
#CUDA_OBJECTS+=$(GELIB_CUDADIR)/SO3partA_CGproduct.o $(GELIB_CUDADIR)/SO3partA_DiagCGproduct.o \
$(CNINE_ROOT)/cuda/CtensorA_add.o 
CUDA_OBJECTS+=$(GELIB_CUDADIR)/GElib_base.o 
CUDA_OBJECTS+=$(GELIB_CUDADIR)/SO3partB_addCGproduct.o $(GELIB_CUDADIR)/SO3partB_addCGproduct_back0.o $(GELIB_CUDADIR)/SO3partB_addCGproduct_back1.o 
CUDA_OBJECTS+=$(GELIB_CUDADIR)/SO3partB_addDiagCGproduct.o $(GELIB_CUDADIR)/SO3partB_addDiagCGproduct_back0.o $(GELIB_CUDADIR)/SO3partB_addDiagCGproduct_back1.o 
CUDA_OBJECTS+=$(GELIB_CUDADIR)/SO3Fpart_addFproduct.o $(GELIB_CUDADIR)/SO3Fpart_addFproduct_back0.o $(GELIB_CUDADIR)/SO3Fpart_addFproduct_back1.o 
CUDA_OBJECTS+=$(GELIB_CUDADIR)/SO3part_addCGtransform.o 
CUDA_EXTERNS+=$(CNINE_ROOT)/include/Cnine_base.cu 
endif 


$(TESTS): %: %.cpp $(DEPS)
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
	rm -f $(TESTS) $(EXECS) $(OBJECTS) 

anew: clean all

