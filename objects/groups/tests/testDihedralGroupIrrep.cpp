#include "GElib_base.cpp"
#include "GElibSession.hpp"

#include "RtensorObj.hpp"
#include "CtensorObj.hpp"
#include "GroupClasses.hpp"

using namespace cnine;
using namespace GElib;

//typedef CscalarObj cscalar;
typedef RtensorObj rtensor;
typedef CtensorObj ctensor;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  const int n=5;
  DihedralGroup G(n);

  DihedralGroupIrrep rho=G.irrep(2);
  for(int i=0; i<2*n; i++) 
    cout<<rho(i)<<endl;

}

