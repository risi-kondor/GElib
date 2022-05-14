#include "GElib_base.cpp"
#include "SO3vecB.hpp"
#include "GElibSession.hpp"


using namespace cnine;
using namespace GElib;

typedef CtensorB ctensor;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  int b=2;
  int L=4;

  SO3vecB v=SO3vecB::Fgaussian(b,L);
  print(v);

  CtensorB A=SO3_iFFT(v,5,5,5);
  print(A);

  SO3vecB w=SO3_FFT(A,L);
  print(w);

}
