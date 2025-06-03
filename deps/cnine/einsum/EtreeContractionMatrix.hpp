#ifndef _EtreeContractionMatrix
#define _EtreeContractionMatrix

namespace cnine{

  class EtreeContractionMatrix{
  public:

    int tensor_ix;


  public: // ---- I/O ----------------------------------------------------------------------------------------

    string cpu_code(const EtreeParams& params, int code_depth){
      CNINE_ASSRT(chidren.size()==1);
      ostringstream oss;
      oss<<string(2*code_depth,' ')<<"TensorView<float > T"<<to_string(tensor_ix)<<"(dims("<<params.dims(indices)<<",0,0);"<<endl;
      oss<<children[0]->cpu_code(params,code_depth+1);
      return oss.str();
    }

  };

  //(A_{ij}*B_{jk})*C_{kl}
  inline dummy_fn(){
    for(int i=0; i<I; i++){
      TensorView<float> T0({K});
      for(int k=0; k<K; k++){ // note order 
	float t=0;
	for(int j=0; j<J; j++)
	  t+=A(i,j)*B(j,k);
	TO(k)=t;
      }
      for(int l=0; l<L; l++){
	float t=0;
	for(int k=0; k<K; k++)
	  t+=T0(i,k)*C(k,l);
	R(i,l)=t;
	}
      }
  }

}

#endif 


