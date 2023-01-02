#include "GElib_base.cpp"
#include "SO3vecB_array.hpp"
#include "GElibSession.hpp"
#include "SO3BiType.hpp"
#include "SO3BiVec_array.hpp"


using namespace cnine;
using namespace GElib;

typedef CtensorB ctensor;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  SO3BiType tau({{{1,1},3}});
  cout<<tau<<endl;

  SO3BiVec_array A=SO3BiVec_array::gaussian({3},tau);
  cout<<A<<endl;


}

