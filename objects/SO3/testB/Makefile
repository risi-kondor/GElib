ROOTDIR=../../..
include $(ROOTDIR)/common.txt

INCLUDE= $(CNINE_INCLUDES) -I$(GELIB_INCLUDEDIR) -I$(SO3DIR) -I$(SO3DIR)/cell_ops -I$(SO3DIR)/functions   

TESTS= testSO3partB testSO3partB_covariance testSO3partB_covariance2 testSO3vecB testSO3vecB_covariance testSO3vecB_gradient \
timeSO3CG \
#testSO3partB_CGproduct 
TESTS+=testSO3partB_CGproduct_back0 testSO3partB_CGproduct_back1 \
testSO3partB_Fproduct testSO3partB_Fproduct_back0 testSO3partB_Fproduct_back1 \
testSO3vecB_Fproduct testSO3vecB_Fproduct_back0 testSO3vecB_Fproduct_back1 \
testSO3vecB_Fmodsq \
testSO3vecB_CGsquare \
testSO3vecB_BlockedCGproduct 

#testSO3Fvec testSO3Fvec_covariance testSO3Fvec_Fmodsq

ifdef WITH_CUDA
TESTS+=
endif 

DEPS=$(SO3DIR)/*.hpp 

EXECS= 

CUDA_OBJECTS=  
CUDA_EXTERNS=

ifdef WITH_CUDA
INCLUDE+=-I$(CUDA_HOME)/include
#CUDA_OBJECTS+=$(GELIB_CUDADIR)/SO3partA_CGproduct.o $(GELIB_CUDADIR)/SO3partA_DiagCGproduct.o \
#$(CNINE_ROOT)/cuda/CtensorA_add.o 
CUDA_OBJECTS+=$(GELIB_CUDADIR)/GElib_base.o 
CUDA_OBJECTS+=$(GELIB_CUDADIR)/SO3partB_addCGproduct.o $(GELIB_CUDADIR)/SO3partB_addCGproduct_back0.o $(GELIB_CUDADIR)/SO3partB_addCGproduct_back1.o 
CUDA_OBJECTS+=$(GELIB_CUDADIR)/SO3partB_addDiagCGproduct.o $(GELIB_CUDADIR)/SO3partB_addDiagCGproduct_back0.o $(GELIB_CUDADIR)/SO3partB_addDiagCGproduct_back1.o 
CUDA_OBJECTS+=$(GELIB_CUDADIR)/SO3Fpart_addFproduct.o $(GELIB_CUDADIR)/SO3Fpart_addFproduct_back0.o $(GELIB_CUDADIR)/SO3Fpart_addFproduct_back1.o 
CUDA_EXTERNS+=$(CNINE_ROOT)/include/Cnine_base.cu 
endif 



# -------------------------------------------------------------------------------------------------------------


testSO3partB: testSO3partB.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3partB testSO3partB.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3partB testSO3partB.cpp $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
endif


testSO3partB_covariance: testSO3partB_covariance.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3partB_covariance testSO3partB_covariance.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3partB_covariance testSO3partB_covariance.cpp $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
endif


testSO3partB_covariance2: testSO3partB_covariance2.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3partB_covariance2 testSO3partB_covariance2.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3partB_covariance2 testSO3partB_covariance2.cpp $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
endif


testSO3vecB: testSO3vecB.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3vecB testSO3vecB.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3vecB testSO3vecB.cpp $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
endif


testSO3vecB_covariance: testSO3vecB_covariance.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3vecB_covariance testSO3vecB_covariance.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3vecB_covariance testSO3vecB_covariance.cpp $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
endif


testSO3vecB_gradient: testSO3vecB_gradient.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3vecB_gradient testSO3vecB_gradient.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3vecB_gradient testSO3vecB_gradient.cpp $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
endif

timeSO3CG: timeSO3CG.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o timeSO3CG timeSO3CG.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o timeSO3CG timeSO3CG.cpp $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
endif



testSO3partB_CGproduct: testSO3partB_CGproduct.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3partB_CGproduct testSO3partB_CGproduct.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3partB_CGproduct testSO3partB_CGproduct.cpp $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
endif

testSO3partB_CGproduct_back0: testSO3partB_CGproduct_back0.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3partB_CGproduct_back0 testSO3partB_CGproduct_back0.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3partB_CGproduct_back0 testSO3partB_CGproduct_back0.cpp $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
endif

testSO3partB_CGproduct_back1: testSO3partB_CGproduct_back1.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3partB_CGproduct_back1 testSO3partB_CGproduct_back1.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3partB_CGproduct_back1 testSO3partB_CGproduct_back1.cpp $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
endif



testSO3partB_Fproduct: testSO3partB_Fproduct.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3partB_Fproduct testSO3partB_Fproduct.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3partB_Fproduct testSO3partB_Fproduct.cpp $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
endif

testSO3partB_Fproduct_back0: testSO3partB_Fproduct_back0.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3partB_Fproduct_back0 testSO3partB_Fproduct_back0.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3partB_Fproduct_back0 testSO3partB_Fproduct_back0.cpp $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
endif

testSO3partB_Fproduct_back1: testSO3partB_Fproduct_back1.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3partB_Fproduct_back1 testSO3partB_Fproduct_back1.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3partB_Fproduct_back1 testSO3partB_Fproduct_back1.cpp $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
endif



testSO3vecB_Fproduct: testSO3vecB_Fproduct.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3vecB_Fproduct testSO3vecB_Fproduct.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3vecB_Fproduct testSO3vecB_Fproduct.cpp $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
endif

testSO3vecB_Fproduct_back0: testSO3vecB_Fproduct_back0.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3vecB_Fproduct_back0 testSO3vecB_Fproduct_back0.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3vecB_Fproduct_back0 testSO3vecB_Fproduct_back0.cpp $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
endif

testSO3vecB_Fproduct_back1: testSO3vecB_Fproduct_back1.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3vecB_Fproduct_back1 testSO3vecB_Fproduct_back1.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3vecB_Fproduct_back1 testSO3vecB_Fproduct_back1.cpp $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
endif

timeSO3vecB_Fproduct: timeSO3vecB_Fproduct.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o timeSO3vecB_Fproduct timeSO3vecB_Fproduct.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o timeSO3vecB_Fproduct timeSO3vecB_Fproduct.cpp $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
endif



testSO3vecB_Fmodsq: testSO3vecB_Fmodsq.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3vecB_Fmodsq testSO3vecB_Fmodsq.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3vecB_Fmodsq testSO3vecB_Fmodsq.cpp $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
endif

testSO3vecB_Fmodsq_back: testSO3vecB_Fmodsq_back.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3vecB_Fmodsq_back testSO3vecB_Fmodsq_back.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3vecB_Fmodsq_back testSO3vecB_Fmodsq_back.cpp $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
endif


testSO3vecB_CGsquare: testSO3vecB_CGsquare.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3vecB_CGsquare testSO3vecB_CGsquare.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3vecB_CGsquare testSO3vecB_CGsquare.cpp $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
endif



testSO3vecB_BlockedCGproduct: testSO3vecB_BlockedCGproduct.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3vecB_BlockedCGproduct testSO3vecB_BlockedCGproduct.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3vecB_BlockedCGproduct testSO3vecB_BlockedCGproduct.cpp $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
endif






# -------------------------------------------------------------------------------------------------------------



objects: $(OBJECTS) $(CUDA_OBJECTS)
tests: $(TESTS)
execs: $(EXECS)
derived: $(DERIVED) 

all: objects execs derived tests 

clean: 
	rm -f $(TESTS) $(EXECS) $(OBJECTS) 

anew: clean all



