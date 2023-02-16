ROOTDIR=../../..
include $(ROOTDIR)/common.txt

INCLUDE= $(CNINE_INCLUDES) -I$(GELIB_INCLUDEDIR) -I$(SO3DIR) -I$(SO3DIR)/cell_ops -I$(SO3DIR)/functions   
INCLUDE+= -I$(ROOTDIR)/objects/Legendre

TESTS=$(patsubst %.cpp,%,$(wildcard *.cpp))

#DEPS=$(TENSORDIR)/*.hpp $(INCLUDEDIR)/*.hpp $(SCALARDIR)/*.hpp $(BACKENDADIR)/*.hpp $(BACKENDBDIR)/*.hpp 
DEPS=$(ROOTDIR)/objects/Legendre/*.hpp 

OBJECTS= 

ifdef WITH_CUDA
INCLUDE+=-I$(CUDA_HOME)/include
CUDA_EXTERNS+=$(INCLUDEDIR)/Cnine_base.cu 
CUDA_OBJECTS=
CUDA_OBJECTS+=$(CUDADIR)/RtensorUtils.o
endif 

ifdef EIGENDIR
OBJECTS+=$(CUDADIR)/EigenRoutines.o
endif 

$(TESTS): %: %.cpp $(DEPS)
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o $@ $@.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o $@ $@.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


tests: $(TESTS)
execs: $(EXECS)

all: execs tests 

clean: 
	rm -f $(TESTS) $(EXECS) 

anew: clean all
