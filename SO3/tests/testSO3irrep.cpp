#include "GElib_base.cpp"
#include "SO3irrep.hpp"

using namespace cnine;
using namespace GElib;


int main(int argc, char** argv){
  cout<<endl;

  SO3element<float> E=SO3element<float>::identity();
  SO3element<float> R0=SO3element<float>::random();
  SO3element<float> R1=SO3element<float>::random();
  SO3element<float> R2=R0*R1;

  for(int l=0; l<3; l++){
    SO3irrep rho(l);
    cout<<rho.matrix(E)<<endl;
  }

  for(int l=0; l<3; l++){
    SO3irrep rho(l);
    cout<<rho.matrix(R0)<<endl;
  }

  for(int l=0; l<3; l++){
    SO3irrep rho(l);

    auto A=rho.matrix(R0);
    auto B=rho.matrix(R1);
    auto C=rho.matrix(R2);
    cout<<A*B<<endl;
    cout<<C<<endl;
  }


}
