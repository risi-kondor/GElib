ROOTDIR=../../..
include $(ROOTDIR)/common.txt

INCLUDE= $(CNINE_INCLUDES) -I$(GELIB_INCLUDEDIR) -I$(SO3DIR) -I$(SO3DIR)/cell_ops -I$(SO3DIR)/functions   
INCLUDE+=-I$(SO2DIR) -I$(SO2DIR)/functions 

TESTS= testSO3part testSO3part2 
#TESTS+= testSO3part_CGproduct_covariance
TESTS+= testSO3vec #testSO3vec2 testSO3vec3 
#TESTS+= testSpharm\

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
CUDA_OBJECTS+= #$(GELIB_CUDADIR)/GElib_base.o 
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





# -------------------------------------------------------------------------------------------------------------



objects: $(OBJECTS) $(CUDA_OBJECTS)
tests: $(TESTS)
execs: $(EXECS)
derived: $(DERIVED) 

all: objects execs derived tests 

clean: 
	rm -f $(TESTS) $(EXECS) $(OBJECTS) 

anew: clean all


