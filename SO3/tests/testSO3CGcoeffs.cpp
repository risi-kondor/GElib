#include "GElib_base.cpp"


using namespace cnine;
using namespace GElib;

int main(int argc, char** argv){
  //GElibSession session;
  cout<<endl;

  //SO3_CGcoeffs<float> C(CGindex(1,1,1));
  //print(C);

  auto CGmatrix=SO3_CGbank.get<float>(1,1,1);
  cout<<CGmatrix<<endl;

}
