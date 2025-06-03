EIGENDIR=/usr/local/include

ROOTDIR=../..
include $(ROOTDIR)/common.txt

INCLUDE= $(CNINE_INCLUDES)
INCLUDE+= $(CENGINE_INCLUDES)

TESTS=

#$(patsubst %.cpp,%,$(wildcard *.cpp))

DEPS=$(TENSORDIR)/*.hpp $(TENSORDIR)/functions/*.hpp 

EXECS=MakeEinsum1Functions 

OBJECTS=Einsum1_loops

CUDA_OBJECTS=  
CUDA_EXTERNS=

ifdef WITH_CUDA
INCLUDE+=-I$(CUDA_HOME)/include
#CUDA_OBJECTS+=$(CETENSORDIR)/CtensorBpack.o 
CUDA_EXTERNS+=$(INCLUDEDIR)/Cnine_base.cu 
endif 


# -------------------------------------------------------------------------------------------------------------


$(EXECS): %: %.cpp
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o $@ $@.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o $@ $@.cpp $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
endif


$(OBJECTS): %: %.cpp
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -c  $@.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE)
else
	$(CC) -c $@.cpp $(CFLAGS) $(MACROS) $(INCLUDE)
endif


tests: $(TESTS)
execs: $(EXECS)
objects: $(OBJECTS)

all: tests 

clean: 
	rm -f $(TESTS) 

anew: clean all
