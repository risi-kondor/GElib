ROOTDIR=../../..
include $(ROOTDIR)/common.txt

INCLUDE= $(CNINE_INCLUDES)
INCLUDE+= -I$(BACKENDBDIR)/cell_ops 
#INCLUDE= -I$(INCLUDEDIR) -I$(SCALARDIR) -I$(TENSORDIR) 
#INCLUDE+= -I$(TENSORARRAYDIR) -I$(TENSORARRAYDIR)/cell_maps -I$(TENSORARRAYDIR)/cell_ops 
#INCLUDE+= -I$(TENSORVIEWDIR) -I$(TENSORVIEWDIR)/ops -I$(TENSORVIEWDIR)/functions  
#INCLUDE+= -I$(BACKENDADIR) -I$(BACKENDADIR)/cell_ops 
#INCLUDE+= -I$(BACKENDBDIR) -I$(BACKENDBDIR)/cell_ops


#TESTS=$(patsubst %.cpp,%,$(wildcard *.cpp))

TESTS= testCtensorArray
TESTS+= testCtensorArray2
TESTS+= testCtensorArray_access
TESTS+= testCtensorArray_accumulator
TESTS+= testCtensorArray_broadcast
TESTS+= testCtensorArray_constructors
TESTS+= testCtensorArray_copy_kernel
TESTS+= testCtensorArray_devices
TESTS+= testCtensorArray_plus_kernel
TESTS+= testCtensorArray_reshape
TESTS+= testRtensorArray

DEPS=$(TENSORDIR)/*.hpp $(INCLUDEDIR)/*.hpp $(SCALARDIR)/*.hpp $(TENSORARRAYDIR)/*.hpp 
DEPS+=$(TENSORARRAYDIR)/cell_ops/*.hpp  

EXECS= 

CUDA_OBJECTS=  
CUDA_EXTERNS=

ifdef WITH_CUDA
INCLUDE+=-I$(CUDA_HOME)/include
CUDA_OBJECTS+=$(CNINE_CUDADIR)/CtensorA_add.o $(CNINE_CUDADIR)/CtensorA_add_plus.o \
$(CNINE_CUDADIR)/CtensorA_copy.o $(CNINE_CUDADIR)/CtensorA_plus.o 

CUDA_EXTERNS+=$(INCLUDEDIR)/Cnine_base.cu 
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
	rm -f $(TESTS) $(EXECS) $(OBJECTS) $(CUDA_OBJECTS) 

anew: clean all





