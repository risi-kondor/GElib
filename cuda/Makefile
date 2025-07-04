ROOTDIR=..
include $(ROOTDIR)/common.txt

INCLUDE= $(CNINE_INCLUDES) 
INCLUDE+=-I$(INCLUDEDIR) -I$(COREDIR) -I$(SO3DIR) -I$(SO3DIR)/functions   

DEPS=$(SO3DIR)/*.hpp 

CUDA_OBJECTS=  
CUDA_EXTERNS=

ifdef WITH_CUDA
INCLUDE+=-I$(CUDA_HOME)/include
CUDA_OBJECTS+=GElib_base.o 
CUDA_OBJECTS+=SO3part_addCGproduct.o SO3part_addCGproduct_back0.o SO3part_addCGproduct_back1.o 
CUDA_EXTERNS+=$(CNINE_ROOT)/include/Cnine_base.cu 
endif 




# -------------------------------------------------------------------------------------------------------------


$(CUDA_OBJECTS): %.o: 
	$(NVCC) $(NVCCFLAGS) -c $*.cu $(CFLAGS) $(MACROS) $(INCLUDE) 


objects: $(OBJECTS) $(CUDA_OBJECTS)

all: objects 

clean: 
	rm -f $(OBJECTS) $(CUDA_OBJECTS) 

anew: clean all
