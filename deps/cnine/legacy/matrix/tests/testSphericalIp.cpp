#include "Cnine_base.cpp"
#include "RtensorA.hpp"
#include "CnineSession.hpp"
#include "SphericalIpMatrix.hpp"

using namespace cnine;

typedef RtensorA rtensor; 


int main(int argc, char** argv){

  cnine_session session;
  cout<<endl;

  int n=5;

  SphericalIpMatrix<float> M(5,10,20,10);
  cout<<M<<endl;

}
