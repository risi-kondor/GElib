ROOTDIR=..
include $(ROOTDIR)/common.txt

INCLUDE= $(CNINE_INCLUDES) -I$(GELIB_INCLUDEDIR) -I$(SO3DIR) -I$(SO3DIR)/cell_ops  

TESTS= 

ifdef WITH_CUDA
TESTS+= 
endif 

DEPS=$(SO3DIR)/*.hpp 

EXECS= 

CUDA_OBJECTS=  
CUDA_EXTERNS=

ifdef WITH_CUDA
INCLUDE+=-I$(CUDA_HOME)/include
CUDA_OBJECTS+=GElib_base.o
CUDA_OBJECTS+=SO3partA_CGproduct.o SO3partA_DiagCGproduct.o
CUDA_OBJECTS+=SO3partB_addCGproduct.o SO3partB_addCGproduct_back0.o SO3partB_addCGproduct_back1.o 
CUDA_OBJECTS+=SO3Fpart_addFmodsq.o SO3Fpart_addFmodsq_back0.o SO3Fpart_addFmodsq_back1.o 
CUDA_EXTERNS+=$(CNINE_ROOT)/include/Cnine_base.cu 
endif 




# -------------------------------------------------------------------------------------------------------------


GElib_base.o: GElib_base.cu Makefile  
	$(NVCC) $(NVCCFLAGS) -c GElib_base.cu $(CFLAGS) $(MACROS) $(INCLUDE) 

SO3partA_CGproduct.o: SO3partA_CGproduct.cu Makefile  
	$(NVCC) $(NVCCFLAGS) -c SO3partA_CGproduct.cu $(CFLAGS) $(MACROS) $(INCLUDE) 

SO3partA_DiagCGproduct.o: SO3partA_DiagCGproduct.cu Makefile  
	$(NVCC) $(NVCCFLAGS) -c SO3partA_DiagCGproduct.cu $(CFLAGS) $(MACROS) $(INCLUDE) 

SO3partB_addCGproduct.o: SO3partB_addCGproduct.cu Makefile  
	$(NVCC) $(NVCCFLAGS) -c SO3partB_addCGproduct.cu $(CFLAGS) $(MACROS) $(INCLUDE) 

SO3partB_addCGproduct_back0.o: SO3partB_addCGproduct_back0.cu Makefile  
	$(NVCC) $(NVCCFLAGS) -c SO3partB_addCGproduct_back0.cu $(CFLAGS) $(MACROS) $(INCLUDE) 

SO3partB_addCGproduct_back1.o: SO3partB_addCGproduct_back1.cu Makefile  
	$(NVCC) $(NVCCFLAGS) -c SO3partB_addCGproduct_back1.cu $(CFLAGS) $(MACROS) $(INCLUDE) 


SO3Fpart_addFmodsq.o: SO3Fpart_addFmodsq.cu Makefile  
	$(NVCC) $(NVCCFLAGS) -c SO3Fpart_addFmodsq.cu $(CFLAGS) $(MACROS) $(INCLUDE) 

SO3Fpart_addFmodsq_back0.o: SO3Fpart_addFmodsq_back0.cu Makefile  
	$(NVCC) $(NVCCFLAGS) -c SO3Fpart_addFmodsq_back0.cu $(CFLAGS) $(MACROS) $(INCLUDE) 

SO3Fpart_addFmodsq_back1.o: SO3Fpart_addFmodsq_back1.cu Makefile  
	$(NVCC) $(NVCCFLAGS) -c SO3Fpart_addFmodsq_back1.cu $(CFLAGS) $(MACROS) $(INCLUDE) 



objects: $(OBJECTS) $(CUDA_OBJECTS)
tests: $(TESTS)
execs: $(EXECS)
derived: $(DERIVED) 

all: objects execs derived tests 

clean: 
	rm -f $(TESTS) $(EXECS) $(OBJECTS) $(CUDA_OBJECTS) 

anew: clean all