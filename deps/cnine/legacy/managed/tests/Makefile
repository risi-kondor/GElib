ROOTDIR=../../..
include $(ROOTDIR)/common.txt

INCLUDE= $(CNINE_INCLUDES)
INCLUDE+= $(CENGINE_INCLUDES)
INCLUDE+= -I../

#INCLUDE= -I$(INCLUDEDIR) -I$(COMBIDIR) -I$(CONTAINERSDIR) -I$(MATHDIR) -I$(WRAPPERSDIR)  
#INCLUDE+= -I$(SCALARDIR) -I$(MATRIXDIR) 
#INCLUDE+= -I$(TENSORDIR) -I$(NTENSORDIR) -I$(NTENSORDIR)/functions  
#INCLUDE+= -I$(TENSORVIEWDIR) -I$(TENSORVIEWDIR)/functions 
#INCLUDE+= -I$(BACKENDADIR) -I$(LABELEDDIR)

TESTS=$(patsubst %.cpp,%,$(wildcard *.cpp))

DEPS=$(INCLUDEDIR)/*.hpp $(SCALARDIR)/*.hpp  -I$(BACKENDBDIR)/*.hpp 

EXECS=

CUDA_OBJECTS=  
CUDA_EXTERNS=

ifdef WITH_CUDA
INCLUDE+=-I$(CUDA_HOME)/include
#CUDA_OBJECTS+=$(CETENSORDIR)/CtensorBpack.o 
CUDA_EXTERNS+=$(INCLUDEDIR)/Cnine_base.cu 
endif 


# -------------------------------------------------------------------------------------------------------------


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

