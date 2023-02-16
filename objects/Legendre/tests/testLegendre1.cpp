#include "Cnine_base.cpp"
#include "LegendreTransform.hpp"
#include "CnineSession.hpp"

using namespace cnine;
using namespace GElib;

typedef RtensorA rtensor;

int main(int argc, char** argv){

  cnine_session session;

  rtensor x=rtensor::gaussian({10});

  LegendreTransform LT(5,1);
  auto y=LT(x);
  cout<<y<<endl;

  auto yy=LT.apply(x);
  cout<<yy<<endl;

  //rtensor U({{1,2},{3,6}}); 
  //cout<<U<<endl;

}
