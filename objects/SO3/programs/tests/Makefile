ROOTDIR=../../../..
include $(ROOTDIR)/common.txt

INCLUDE= $(CNINE_INCLUDES) -I$(GELIB_INCLUDEDIR) -I$(SO3DIR) -I$(SO3DIR)/cell_ops -I$(SO3DIR)/functions \
-I$(SO3DIR)/programs    

TESTS= testSO3CGprogram1

ifdef WITH_CUDA
TESTS+=
endif 

DEPS=$(SO3DIR)/*.hpp $(SO3DIR)/programs/*.hpp 

EXECS= 

CUDA_OBJECTS=  
CUDA_EXTERNS=

ifdef WITH_CUDA
INCLUDE+=-I$(CUDA_HOME)/include
CUDA_OBJECTS+= #$(GELIB_CUDADIR)/GElib_base.o 
CUDA_OBJECTS+=$(GELIB_CUDADIR)/SO3partB_addCGproduct.o $(GELIB_CUDADIR)/SO3partB_addCGproduct_back0.o $(GELIB_CUDADIR)/SO3partB_addCGproduct_back1.o 
CUDA_OBJECTS+=$(GELIB_CUDADIR)/SO3Fpart_addFproduct.o $(GELIB_CUDADIR)/SO3Fpart_addFproduct_back0.o $(GELIB_CUDADIR)/SO3Fpart_addFproduct_back1.o 
CUDA_EXTERNS+=$(CNINE_ROOT)/include/Cnine_base.cu 
endif 



# -------------------------------------------------------------------------------------------------------------


testSO3CGprogram1: testSO3CGprogram1.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3CGprogram1 testSO3CGprogram1.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3CGprogram1 testSO3CGprogram1.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif




objects: $(OBJECTS) $(CUDA_OBJECTS)
tests: $(TESTS)
execs: $(EXECS)
derived: $(DERIVED) 

all: objects execs derived tests 

clean: 
	rm -f $(TESTS) $(EXECS) $(OBJECTS) 

anew: clean all


