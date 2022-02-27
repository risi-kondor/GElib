ROOTDIR=../../..
include $(ROOTDIR)/common.txt

INCLUDE= $(CNINE_INCLUDES) -I$(GELIB_INCLUDEDIR) -I$(SO3DIR) -I$(SO3DIR)/cell_ops  

TESTS= testSO3partD testSO3vecD 

ifdef WITH_CUDA
TESTS+= 
endif 

DEPS=$(SO3DIR)/*.hpp 

EXECS= 

CUDA_OBJECTS=  
CUDA_EXTERNS=

ifdef WITH_CUDA
INCLUDE+=-I$(CUDA_HOME)/include
CUDA_OBJECTS+=$(GELIB_CUDADIR)/SO3partA_CGproduct.o $(GELIB_CUDADIR)/SO3partA_DiagCGproduct.o \
$(CNINE_ROOT)/cuda/CtensorA_add.o 
CUDA_OBJECTS+=$(GELIB_CUDADIR)/GElib_base.o 
CUDA_OBJECTS+=$(GELIB_CUDADIR)/SO3partB_addCGproduct.o $(GELIB_CUDADIR)/SO3partB_addCGproduct_back0.o $(GELIB_CUDADIR)/SO3partB_addCGproduct_back1.o 
CUDA_OBJECTS+=$(GELIB_CUDADIR)/SO3Fpart_addFproduct.o $(GELIB_CUDADIR)/SO3Fpart_addFproduct_back0.o $(GELIB_CUDADIR)/SO3Fpart_addFproduct_back1.o 
CUDA_EXTERNS+=$(CNINE_ROOT)/include/Cnine_base.cu 
endif 



# -------------------------------------------------------------------------------------------------------------


testSO3partD: testSO3partD.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3partD testSO3partD.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3partD testSO3partD.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


#testSO3partD_covariance: testSO3partD_covariance.cpp $(DEPS) 
#ifdef WITH_CUDA
#	$(NVCC) $(NVCCFLAGS) -o testSO3partD_covariance testSO3partD_covariance.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
#else
#	$(CC) -o testSO3partD_covariance testSO3partD_covariance.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
#endif


testSO3vecD: testSO3vecD.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3vecD testSO3vecD.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3vecD testSO3vecD.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


#testSO3vecD_covariance: testSO3vecD_covariance.cpp $(DEPS) 
#ifdef WITH_CUDA
#	$(NVCC) $(NVCCFLAGS) -o testSO3vecD_covariance testSO3vecD_covariance.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
#else
#	$(CC) -o testSO3vecD_covariance testSO3vecD_covariance.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
#endif


#testSO3vecD_gradient: testSO3vecD_gradient.cpp $(DEPS) 
#ifdef WITH_CUDA
#	$(NVCC) $(NVCCFLAGS) -o testSO3vecD_gradient testSO3vecD_gradient.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
#else
#	$(CC) -o testSO3vecD_gradient testSO3vecD_gradient.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
#endif



# -------------------------------------------------------------------------------------------------------------



objects: $(OBJECTS) $(CUDA_OBJECTS)
tests: $(TESTS)
execs: $(EXECS)
derived: $(DERIVED) 

all: objects execs derived tests 

clean: 
	rm -f $(TESTS) $(EXECS) $(OBJECTS) 

anew: clean all





testSO3partView: testSO3partView.cpp ../*.hpp $(DEPS)
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3partView testSO3partView.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3partView testSO3partView.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


spharm-test: spharm-test.cpp ../*.hpp $(DEPS)
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o spharm-test spharm-test.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o spharm-test spharm-test.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


CGproduct-covariance-test: CGproduct-covariance-test.cpp ../*.hpp $(DEPS)
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o CGproduct-covariance-test CGproduct-covariance-test.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o CGproduct-covariance-test CGproduct-covariance-test.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif

