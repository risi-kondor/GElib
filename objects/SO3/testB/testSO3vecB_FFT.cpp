#include "GElib_base.cpp"
#include "SO3vecB.hpp"
#include "GElibSession.hpp"


using namespace cnine;
using namespace GElib;

typedef CtensorB ctensor;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  int b=1;
  int L=4;
  int N=15;

  SO3vecB v=SO3vecB::Fgaussian(b,L);
  //printl("v=",v);

  for(int i=0; i<10; i++){

  CtensorB A=SO3_iFFT(v,2*N,N,2*N);
  //printl("A=",A);

  SO3vecB w=SO3_FFT(A,L);
  //printl("w",w);
  }

  cout<<"Ratio:"<<endl;
  //print(w/v);


  #ifdef _WITH_CUDA

  SO3vecB vc=v.to_device(1);
  //printl("vc=",vc);
  CtensorB Ac=SO3_iFFT(vc,2*N,N,2*N);
  //printl("Ac=",Ac);
  SO3vecB wc=SO3_FFT(Ac,L);
  //printl("wc=",wc);
  //cout<<"devic="<<wc.get_dev()<<endl;

  SO3vecB vcc=vc.to_device(0);
  SO3vecB wcc=wc.to_device(0);
  cout<<"Ratio:"<<endl;
  print(wcc/vcc);
  
  #endif

  //CtensorB f=CtensorB::gaussian({1,8,4,8});
  //SO3vecB F=SO3_FFT(f,12);
  //CtensorB fd=SO3_iFFT(F,8,4,8);
  //print(fd/f);

}
