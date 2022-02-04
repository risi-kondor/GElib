ROOTDIR=../../..
include $(ROOTDIR)/common.txt

INCLUDE= $(CNINE_INCLUDES) -I$(GELIB_INCLUDEDIR) -I$(SO3DIR) -I$(SO3DIR)/cell_ops  

TESTS= testSO3part testSO3part2 testSO3part_CGproduct_covariance\
testSO3vec testSO3vec2 testSO3vec3 \
testSpharm\
testSO3partArray testSO3partArray_constructors testSO3partArray_broadcast testSO3partArray_access\
testSO3partArray_shape testSO3partArray_arithmetic testSO3partArray_functions \
testSO3partArray_CGproduct testSO3partArray_CGproduct_back0 testSO3partArray_CGproduct_back1 \
testSO3partArray_CGproduct_big testSO3partArray_CGproduct_back0_big testSO3partArray_CGproduct_back1_big \
testSO3partArray_CGproduct_cmaps testSO3partArray_CGproduct_covariance \
testSO3partArray_DiagCGproduct \
testSO3vecArray testSO3vecArray2 testSO3vecArray_constructors testSO3vecArray_broadcast \
testSO3partB testSO3partB_covariance testSO3vecB testSO3vecB_covariance testSO3Fvec testSO3Fvec_covariance \
testSO3Fvec_Fmodsq

ifdef WITH_CUDA
TESTS+= testSO3partArray_constructorsG testSO3partArray_arithmeticG testSO3partArray_broadcastG
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


testSO3part: testSO3part.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3part testSO3part.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3part testSO3part.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testSO3part2: testSO3part2.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3part2 testSO3part2.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3part2 testSO3part2.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testSO3part_CGproduct_covariance: testSO3part_CGproduct_covariance.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3part_CGproduct_covariance testSO3part_CGproduct_covariance.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3part_CGproduct_covariance testSO3part_CGproduct_covariance.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testSpharm: testSpharm.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSpharm testSpharm.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSpharm testSpharm.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif



# -------------------------------------------------------------------------------------------------------------


testSO3vec: testSO3vec.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3vec testSO3vec.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3vec testSO3vec.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testSO3vec2: testSO3vec2.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3vec2 testSO3vec2.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3vec2 testSO3vec2.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testSO3vec3: testSO3vec3.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3vec3 testSO3vec3.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3vec3 testSO3vec3.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


# -------------------------------------------------------------------------------------------------------------


testSO3partArray: testSO3partArray.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3partArray testSO3partArray.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3partArray testSO3partArray.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testSO3partArray_constructors: testSO3partArray_constructors.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3partArray_constructors testSO3partArray_constructors.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3partArray_constructors testSO3partArray_constructors.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testSO3partArray_constructorsG: testSO3partArray_constructorsG.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3partArray_constructorsG testSO3partArray_constructorsG.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3partArray_constructorsG testSO3partArray_constructorsG.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testSO3partArray_broadcastG: testSO3partArray_broadcastG.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3partArray_broadcastG testSO3partArray_broadcastG.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3partArray_broadcastG testSO3partArray_broadcastG.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testSO3partArray_broadcast: testSO3partArray_broadcast.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3partArray_broadcast testSO3partArray_broadcast.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3partArray_broadcast testSO3partArray_broadcast.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testSO3partArray_access: testSO3partArray_access.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3partArray_access testSO3partArray_access.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3partArray_access testSO3partArray_access.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testSO3partArray_shape: testSO3partArray_shape.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3partArray_shape testSO3partArray_shape.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3partArray_shape testSO3partArray_shape.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testSO3partArray_arithmetic: testSO3partArray_arithmetic.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3partArray_arithmetic testSO3partArray_arithmetic.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3partArray_arithmetic testSO3partArray_arithmetic.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testSO3partArray_arithmeticG: testSO3partArray_arithmeticG.cpp $(DEPS) 
ifdef WITH_CUDA
# $(NVCC) $(NVCCFLAGS) -o testSO3partArray_arithmeticG testSO3partArray_arithmeticG.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
	$(CC) -o testSO3partArray_arithmeticG testSO3partArray_arithmeticG.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3partArray_arithmeticG testSO3partArray_arithmeticG.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testSO3partArray_functions: testSO3partArray_functions.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3partArray_functions testSO3partArray_functions.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3partArray_functions testSO3partArray_functions.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testSO3partArray_CGproduct: testSO3partArray_CGproduct.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3partArray_CGproduct testSO3partArray_CGproduct.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
#	$(CC) -o testSO3partArray_CGproduct testSO3partArray_CGproduct.cpp $(CUDA_OBJECTS) $(CFLAGS) $(INCLUDE) $(LIBS) -lcublas
else
	$(CC) -o testSO3partArray_CGproduct testSO3partArray_CGproduct.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testSO3partArray_CGproduct_back0: testSO3partArray_CGproduct_back0.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3partArray_CGproduct_back0 testSO3partArray_CGproduct_back0.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3partArray_CGproduct_back0 testSO3partArray_CGproduct_back0.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testSO3partArray_CGproduct_back1: testSO3partArray_CGproduct_back1.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3partArray_CGproduct_back1 testSO3partArray_CGproduct_back1.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3partArray_CGproduct_back1 testSO3partArray_CGproduct_back1.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testSO3partArray_CGproduct_cmaps: testSO3partArray_CGproduct_cmaps.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3partArray_CGproduct_cmaps testSO3partArray_CGproduct_cmaps.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
#	$(CC) -o testSO3partArray_CGproduct testSO3partArray_CGproduct.cpp $(CUDA_OBJECTS) $(CFLAGS) $(INCLUDE) $(LIBS) -lcublas
else
	$(CC) -o testSO3partArray_CGproduct_cmaps testSO3partArray_CGproduct_cmaps.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testSO3partArray_CGproduct_covariance: testSO3partArray_CGproduct_covariance.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3partArray_CGproduct_covariance testSO3partArray_CGproduct_covariance.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
#	$(CC) -o testSO3partArray_CGproduct_covariance testSO3partArray_CGproduct_covariance.cpp $(CUDA_OBJECTS) $(CFLAGS) $(INCLUDE) $(LIBS) -lcublas
else
	$(CC) -o testSO3partArray_CGproduct_covariance testSO3partArray_CGproduct_covariance.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testSO3partArray_DiagCGproduct: testSO3partArray_DiagCGproduct.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3partArray_DiagCGproduct testSO3partArray_DiagCGproduct.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3partArray_DiagCGproduct testSO3partArray_DiagCGproduct.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif



testSO3partArray_CGproduct_big: testSO3partArray_CGproduct_big.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3partArray_CGproduct_big testSO3partArray_CGproduct_big.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3partArray_CGproduct_big testSO3partArray_CGproduct_big.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testSO3partArray_CGproduct_back0_big: testSO3partArray_CGproduct_back0_big.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3partArray_CGproduct_back0_big testSO3partArray_CGproduct_back0_big.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3partArray_CGproduct_back0_big testSO3partArray_CGproduct_back0_big.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testSO3partArray_CGproduct_back1_big: testSO3partArray_CGproduct_back1_big.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3partArray_CGproduct_back1_big testSO3partArray_CGproduct_back1_big.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3partArray_CGproduct_back1_big testSO3partArray_CGproduct_back1_big.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif




# -------------------------------------------------------------------------------------------------------------


testSO3vecArray: testSO3vecArray.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3vecArray testSO3vecArray.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3vecArray testSO3vecArray.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testSO3vecArray2: testSO3vecArray2.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3vecArray2 testSO3vecArray2.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3vecArray2 testSO3vecArray2.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testSO3vecArray_constructors: testSO3vecArray_constructors.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3vecArray_constructors testSO3vecArray_constructors.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3vecArray_constructors testSO3vecArray_constructors.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testSO3vecArray_broadcast: testSO3vecArray_broadcast.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3vecArray_broadcast testSO3vecArray_broadcast.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3vecArray_broadcast testSO3vecArray_broadcast.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testSO3vecArray_CGproduct: testSO3vecArray_CGproduct.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3vecArray_CGproduct testSO3vecArray_CGproduct.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3vecArray_CGproduct testSO3vecArray_CGproduct.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


# -------------------------------------------------------------------------------------------------------------


testSO3partB: testSO3partB.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3partB testSO3partB.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3partB testSO3partB.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testSO3partB_covariance: testSO3partB_covariance.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3partB_covariance testSO3partB_covariance.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3partB_covariance testSO3partB_covariance.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testSO3vecB: testSO3vecB.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3vecB testSO3vecB.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3vecB testSO3vecB.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testSO3vecB_covariance: testSO3vecB_covariance.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3vecB_covariance testSO3vecB_covariance.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3vecB_covariance testSO3vecB_covariance.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testSO3Fvec: testSO3Fvec.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3Fvec testSO3Fvec.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3Fvec testSO3Fvec.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testSO3Fvec_covariance: testSO3Fvec_covariance.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3Fvec_covariance testSO3Fvec_covariance.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3Fvec_covariance testSO3Fvec_covariance.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testSO3Fvec_Fmodsq: testSO3Fvec_Fmodsq.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSO3Fvec_Fmodsq testSO3Fvec_Fmodsq.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSO3Fvec_Fmodsq testSO3Fvec_Fmodsq.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
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

