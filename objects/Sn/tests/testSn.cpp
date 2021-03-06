#include "Snob2_base.cpp"
#include "Snob2_session.hpp"
#include "SnClasses.hpp"


using namespace cnine;
using namespace GElib;

//typedef CscalarObj cscalar;
//typedef CtensorObj ctensor;



int main(int argc, char** argv){
  Snob2_session session;
  SnClasses Snclasses;
  cout<<endl;

  Sn sn(4);
  cout<<sn.get_order()<<endl;

  SnElement sigma=sn.identity(); 
  cout<<sigma<<endl;
  cout<<endl; 

  int N=sn.get_order();
  for(int i=0; i<N; i++)
    cout<<sn.element(i)<<endl;

}
