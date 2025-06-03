ROOTDIR=..
include $(ROOTDIR)/common.txt

INCLUDE= $(CNINE_INCLUDES)

#INCLUDE= -I$(INCLUDEDIR)  -I$(SCALARDIR) -I$(MATRIXDIR) -I$(TENSORDIR) -I$(NTENSORDIR) -I$(TENSORVIEWDIR) -I$(TENSORARRAYDIR) -I$(TENSORARRAYDIR)/cell_ops -I$(TENSORARRAYDIR)/cell_maps  
#INCLUDE+= -I$(LABELEDDIR) 
#INCLUDE+= -I$(BACKENDADIR) -I$(BACKENDBDIR) \
#-I$(MANAGEDDIR) -I$(MSCALARDIR) -I$(MTENSORDIR) \
#$(CENGINE_INCLUDES) 

TESTS=

DEPS= $(TENSORDIR)/*.hpp $(INCLUDEDIR)/*.hpp $(INCLUDEDIR)/cmaps/*.hpp $(SCALARDIR)/*.hpp $(MSCALARDIR)/*.hpp $(MTENSORDIR)/*.hpp

EXECS= 

OBJECTS=
ifdef EIGENDIR
OBJECTS+=EigenRoutines.o 
endif 

CUDA_OBJECTS=  
CUDA_EXTERNS=

ifdef WITH_CUDA
INCLUDE+=-I$(CUDA_HOME)/include
#CUDA_OBJECTS+=Flock.o CtensorA_add.o CtensorA_add_plus.o CtensorA_copy.o CtensorA_plus.o
CUDA_OBJECTS+=TensorView_accumulators.o  
#CUDA_OBJECTS+=BasicCtensorProducts.o  
#CUDA_OBJECTS+=RtensorEinsumProducts.o  
CUDA_OBJECTS+=RtensorUtils.o  
#CUDA_OBJECTS+=RtensorPackUtils.o  
#CUDA_OBJECTS+=RtensorConvolve2d.o 
#CUDA_OBJECTS+=RtensorConvolve2dSparse.o 
#CUDA_OBJECTS+=RtensorConvolve3d.o  
#CUDA_OBJECTS+=RtensorConvolve3dSparse.o  
CUDA_OBJECTS+=gatherRows.o  
CUDA_OBJECTS+=TensorView_add.o  
CUDA_OBJECTS+=TensorView_assign.o  
CUDA_OBJECTS+=TensorView_inc.o  
CUDA_OBJECTS+=TensorView_gather.o  
CUDA_OBJECTS+=BlockCsparseMatrix.o  
CUDA_EXTERNS+=$(INCLUDEDIR)/Cnine_base.cu 
NVCCFLAGS+= -D_WITH_CUDA -D_WITH_CUBLAS
endif 

$(CUDA_OBJECTS): %.o: 
	$(NVCC) $(NVCCFLAGS) -c $*.cu $(CFLAGS) $(MACROS) $(INCLUDE) 


# -------------------------------------------------------------------------------------------------------------



objects: $(OBJECTS) $(CUDA_OBJECTS)
tests: $(TESTS)
execs: $(EXECS)
derived: $(DERIVED) 

all: objects  

clean: 
	rm -f $(CUDA_OBJECTS) 

anew: clean all


