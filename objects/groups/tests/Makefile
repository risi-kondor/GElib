ROOTDIR=../../..
include $(ROOTDIR)/common.txt

INCLUDE= $(CNINE_INCLUDES) -I$(GELIB_INCLUDEDIR) -I$(SO3DIR) -I$(GROUPSDIR) 

TESTS= testCyclicGroup testIrrep \
testDihedralGroup testDihedralGroupIrrep testDihedralGroupFourier \
testProductGroup testProductGroupIrrep \
testGroupAlgebra testFunctionOnGroup testFourierOnGroup 

ifdef WITH_CUDA
endif 

DEPS=$(GROUPSDIR)/*.hpp $(COMBINATORIALDIR)/*.hpp  

EXECS= 

CUDA_OBJECTS=  
CUDA_EXTERNS=

ifdef WITH_CUDA
INCLUDE+=-I$(CUDA_HOME)/include
#CUDA_OBJECTS+=$(GELIB_CUDADIR)/SO3partA_CGproduct.o $(CNINE_ROOT)/cuda/CtensorA_add.o 
CUDA_EXTERNS+=$(CNINE_ROOT)/include/Cnine_base.cu 
endif 



# -------------------------------------------------------------------------------------------------------------


testCyclicGroup: testCyclicGroup.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testCyclicGroup testCyclicGroup.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testCyclicGroup testCyclicGroup.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testIrrep: testIrrep.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testIrrep testIrrep.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testIrrep testIrrep.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif





testDihedralGroup: testDihedralGroup.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testDihedralGroup testDihedralGroup.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testDihedralGroup testDihedralGroup.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testDihedralGroupIrrep: testDihedralGroupIrrep.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testDihedralGroupIrrep testDihedralGroupIrrep.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testDihedralGroupIrrep testDihedralGroupIrrep.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testDihedralGroupFourier: testDihedralGroupFourier.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testDihedralGroupFourier testDihedralGroupFourier.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testDihedralGroupFourier testDihedralGroupFourier.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif




testProductGroup: testProductGroup.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testProductGroup testProductGroup.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testProductGroup testProductGroup.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testProductGroupIrrep: testProductGroupIrrep.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testProductGroupIrrep testProductGroupIrrep.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testProductGroupIrrep testProductGroupIrrep.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif




testGroupAlgebra: testGroupAlgebra.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testGroupAlgebra testGroupAlgebra.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testGroupAlgebra testGroupAlgebra.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testFunctionOnGroup: testFunctionOnGroup.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testFunctionOnGroup testFunctionOnGroup.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testFunctionOnGroup testFunctionOnGroup.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testFourierOnGroup: testFourierOnGroup.cpp $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testFourierOnGroup testFourierOnGroup.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testFourierOnGroup testFourierOnGroup.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
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





