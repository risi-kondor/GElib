#include "GElib_base.cpp"
#include "O3irrep.hpp"

using namespace cnine;
using namespace GElib;


int main(int argc, char** argv){
  cout<<endl;

  O3element<float> R0=O3element<float>::random();
  O3element<float> R1=O3element<float>::random();
  O3element<float> R2=R0*R1;

  for(int l=0; l<3; l++){
    O3irrep rho({l,1});
    cout<<rho.matrix(R0)<<endl;
  }

  for(int l=0; l<3; l++){
    O3irrep rho({l,1});

    auto A=rho.matrix(R0);
    auto B=rho.matrix(R1);
    auto C=rho.matrix(R2);
    cout<<A*B<<endl;
    cout<<C<<endl;
  }


}
