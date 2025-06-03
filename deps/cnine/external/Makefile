ROOTDIR=..
include $(ROOTDIR)/common.txt

INCLUDE= $(CNINE_INCLUDES) 

#INCLUDE= -I$(INCLUDEDIR) -I$(COMBIDIR) -I$(CONTAINERSDIR) -I$(MATHDIR) -I$(WRAPPERSDIR)  
#INCLUDE+= -I$(SCALARDIR) -I$(MATRIXDIR) 
#INCLUDE+= -I$(TENSORDIR) -I$(NTENSORDIR) -I$(NTENSORDIR)/functions  
#INCLUDE+= -I$(TENSORVIEWDIR) -I$(TENSORVIEWDIR)/functions 
#INCLUDE+= -I$(BACKENDADIR) -I$(LABELEDDIR)
#INCLUDE+= -I$(BACKENDADIR) -I$(BACKENDBDIR) 

TESTS=

DEPS= 

EXECS= 

OBJECTS=
ifdef EIGENDIR
OBJECTS+=EigenRoutines 
endif 


# -------------------------------------------------------------------------------------------------------------


$(OBJECTS): %: %.cpp
	$(CC) -c $@.cpp $(CFLAGS) $(MACROS) $(INCLUDE) 


objects: $(OBJECTS) $(CUDA_OBJECTS)
tests: $(TESTS)
execs: $(EXECS)
derived: $(DERIVED) 

all: objects  

clean: 
	rm -f $(CUDA_OBJECTS) 

anew: clean all


