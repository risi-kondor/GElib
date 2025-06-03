#include "Cnine_base.cpp"
#include "RtensorA.hpp"
#include "CnineSession.hpp"
//#include "RtensorConvolve2dFn.hpp"

using namespace cnine;

typedef RtensorA rtensor; 


int main(int argc, char** argv){

  cnine_session session;

  cout<<endl;


  rtensor x=rtensor::gaussian({10,10,1});
  rtensor M=rtensor::gaussian({3,5,5});

  auto y=x.convolve2_prod(M);
  cout<<y<<endl;

  rtensor x3=rtensor::gaussian({10,10,10,1});
  rtensor M3=rtensor::gaussian({3,5,5,5});

  auto y3=x.convolve3_prod(M3);
  cout<<y3<<endl;

}
