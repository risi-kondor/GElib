#include "Cnine_base.cpp"
#include "CnineSession.hpp"
#include "GatherMapB.hpp"
#include "GatherRows.hpp"
#include "GatherSlices.hpp"
#include "Ltensor.hpp"

using namespace cnine;

int main(int argc, char** argv){

  cnine_session session;

  GatherMapB g=GatherMapB::random(5,5,0.4);
  cout<<g<<endl;

  TensorView<float> X(dims(5,3),3,0);
  TensorView<float> Y(dims(5,3),0,0);

  auto Z=GatherSlices()(X,g,0);
  cout<<Z<<endl;

  TensorView<float> Xg(X,1);
  auto Zg=GatherSlices()(Xg,g,0);
  cout<<Zg<<endl;


  if(false){
    TensorView<float> A(dims(3,5,4),4,0);
    auto B=GatherSlices()(A,g,1);
    cout<<B<<endl;
    auto C=GatherSlices().naive(A,g,1);
    cout<<C<<endl;
  }

}

