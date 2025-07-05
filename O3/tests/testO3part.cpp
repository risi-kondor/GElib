#include "GElib_base.cpp"
//#include "GElibSession.hpp"

#include "O3part.hpp"
//#include "Gfunctions.hpp"
#include "CGproduct.hpp"
#include "DiagCGproduct.hpp"

using namespace cnine;
using namespace GElib;


int main(int argc, char** argv){
  //GElibSession session;
  cout<<endl;

  int b=3;
  int l=2;
  int n=2;

  O3part<float> x(O3index(2,1),1,3,4);
  O3part<float> y(O3index(2,1),1,3,4);
  print(x);

  O3part<float> x1(grid=Gdims({2,2}),batch=2,irrep=O3index(2,1),filltype=4,channels=3);
  print(x1);


  cout<<"CG product:"<<endl;
  auto z=CGproduct(x,y,O3index(1,1));
  print(z);

  cout<<"Diag CG product:"<<endl;
  auto w=DiagCGproduct(x,y,O3index(1,1));
  print(w);

}
