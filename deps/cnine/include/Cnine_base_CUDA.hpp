#define CNINE_CONST_MEM_SIZE 32278


#ifdef _WITH_CUDA
#define IFCUDA(cmds) cmds 
#else 
#define IFCUDA(cmds) 
#endif


#ifdef _WITH_CUDA
#define CNINE_REQUIRES_CUDA() 
#else
#define CNINE_REQUIRES_CUDA() printf("Cnine error in \"%s\":  cnine was compiled without CUDA.\n",__PRETTY_FUNCTION__);
#endif 

#ifdef _WITH_CUDA
#define CUDA_SAFE(err) __cudaSafeCall(err, __FILE__, __LINE__ );
inline void __cudaSafeCall(cudaError err, const char *file, const int line){
  if(cudaSuccess!=err){
    fprintf(stderr,"cudaSafeCall() failed at %s:%i : %s\n",file,line,cudaGetErrorString(err));
    exit(-1);}
  return;
}
#else 
#define CUDA_SAFE(err) ; 
#endif 


#ifdef _WITH_CUBLAS
#define CUBLAS_SAFE(expression) {			     \
    cublasStatus_t status= (expression);		     \
    if (status != CUBLAS_STATUS_SUCCESS) {                    \
      std::cerr << "CuBLAS error on line " << __LINE__ << ": ";		\
    if(status==CUBLAS_STATUS_SUCCESS) fprintf(stderr,"CUBLAS SUCCESS"); \
    else if(status==CUBLAS_STATUS_NOT_INITIALIZED) \
        fprintf(stderr,"'CUBLAS_STATUS_NOT_INITIALIZED'"); \
    else if(status==CUBLAS_STATUS_ALLOC_FAILED)\
        fprintf(stderr,"'CUBLAS_STATUS_ALLOC_FAILED'");\
    else if(status==CUBLAS_STATUS_INVALID_VALUE)\
        fprintf(stderr,"'CUBLAS_STATUS_INVALID_VALUE'");\
    else if(status==CUBLAS_STATUS_ARCH_MISMATCH)\
        fprintf(stderr,"'CUBLAS_STATUS_ARCH_MISMATCH'");\
    else if(status==CUBLAS_STATUS_MAPPING_ERROR)\
        fprintf(stderr,"'CUBLAS_STATUS_MAPPING_ERROR'");\
    else if(status==CUBLAS_STATUS_EXECUTION_FAILED)\
        fprintf(stderr,"'CUBLAS_STATUS_EXECUTION_FAILED'");\
    else if(status==CUBLAS_STATUS_INTERNAL_ERROR)\
        fprintf(stderr,"'CUBLAS_STATUS_INTERNAL_ERROR'");\
    else						 \
      fprintf(stderr,"UNKNOWN CUBLAS ERROR");\
    std::exit(EXIT_FAILURE);				     \
    }                                                        \
  }
#else 
#define CUBLAS_SAFE(expression) //expression  
#endif 


#ifdef _WITH_CUDA
#define CUDA_STREAM(cmd)({\
      cudaStream_t stream=NULL;			\
      CUDA_SAFE(cudaStreamCreate(&stream));		\
      cmd;						\
      CUDA_SAFE(cudaStreamSynchronize(stream));\
      CUDA_SAFE(cudaStreamDestroy(stream));\
      })
#else
#define CUDA_STREAM(cmd) CNINE_NOCUDA_ERROR
#endif



#ifdef _WITH_CUDA
class cu_stream{
public:

  cudaStream_t stm;				\

  cu_stream(){
    CUDA_SAFE(cudaStreamCreate(&stm));
  }

  ~cu_stream(){
    CUDA_SAFE(cudaStreamSynchronize(stm));
    CUDA_SAFE(cudaStreamDestroy(stm));
  }

  operator cudaStream_t() const{
    return stm;
  }

};
#endif 

#define CPUCODE(cmd) if(dev==0){cmd;}

#ifdef _WITH_CUDA
#define GPUCODE(cmd) if(dev==1){cmd;}
#else
#define GPUCODE(cmd) 
#endif

namespace cnine{

  template<typename TYPE>
  class ArrayOnDevice{
  public:

    TYPE* arr;

    ArrayOnDevice(const int _n){
      CUDA_SAFE(cudaMalloc((void **)&arr, std::max(_n*sizeof(TYPE),(unsigned long) 1)));
    }

    ~ArrayOnDevice(){
      CUDA_SAFE(cudaFree(arr));
    }

    operator TYPE*() const{
      return arr;
    }
  };

}
