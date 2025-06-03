#include "Cnine_base.cpp"
#include "RtensorA.hpp"
#include "CnineSession.hpp"
#include "InterpolBilinear.hpp"

using namespace cnine;

typedef RtensorA rtensor; 


int main(int argc, char** argv){

  cnine_session session;
  cout<<endl;

  int n=5;

  RtensorA X=RtensorA::zero({1,2});
  X.set(0,0,0.4);
  X.set(0,1,0.5);

  InterpolBilinear<float> M(X,6,6);

  cout<<M<<endl;
}
