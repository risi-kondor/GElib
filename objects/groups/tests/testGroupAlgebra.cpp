#include "GElib_base.cpp"
#include "GElibSession.hpp"

#include "RtensorObj.hpp"
#include "GroupClasses.hpp"

using namespace cnine;
using namespace GElib;

//typedef CscalarObj cscalar;
typedef RtensorObj rtensor;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  CyclicGroup G(5);
  Gdims dims({3});

  GroupAlgebra<CyclicGroup,rtensor> A(G,dims);
  cout<<A<<endl;

}

