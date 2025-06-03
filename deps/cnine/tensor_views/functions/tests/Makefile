ROOTDIR=../../../..
include $(ROOTDIR)/common.txt

INCLUDE= -I$(INCLUDEDIR) -I$(INCLUDEDIR)/cmaps -I$(SCALARDIR) -I$(MATRIXDIR) -I$(TENSORDIR) -I$(TENSORDIR)/ops 
INCLUDE+= -I$(BACKENDADIR) -I$(BACKENDBDIR)
INCLUDE+= -I$(TENSORVIEWDIR) -I$(TENSORVIEWDIR)/ops -I$(TENSORVIEWDIR)/functions 
INCLUDE+= -I$(LABELEDDIR) 
INCLUDE+= -I$(NTENSORDIR) 

TESTS=$(patsubst %.cpp,%,$(wildcard *.cpp))

DEPS=$(TENSORDIR)/*.hpp $(INCLUDEDIR)/*.hpp $(SCALARDIR)/*.hpp  $(TENSORVIEWDIR)/*.hpp 

EXECS= 


CUDA_OBJECTS+=$(CUDADIR)/BasicCtensorProducts.o 
CUDA_OBJECTS+=$(CUDADIR)/RtensorUtils.o  
CUDA_OBJECTS+= $(CUDADIR)/RtensorConvolve2d.o  
CUDA_OBJECTS+= $(CUDADIR)/RtensorConvolve3d.o  
CUDA_OBJECTS+= $(CUDADIR)/RtensorConvolve2dSparse.o  
CUDA_OBJECTS+= $(CUDADIR)/RtensorConvolve3dSparse.o  
CUDA_EXTERNS=

ifdef WITH_CUDA
INCLUDE+=-I$(CUDA_HOME)/include
CUDA_EXTERNS+=$(INCLUDEDIR)/Cnine_base.cu 
endif 

$(TESTS): %: %.cpp
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o $@ $@.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o $@ $@.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif

tests: $(TESTS)

all: tests 

clean: 
	rm -f $(TESTS) 

anew: clean all

