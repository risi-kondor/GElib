//#define DEBUG_ENGINE_FLAG
//#define CENGINE_ECHO_QUEUE

#include "GElib_base.cpp"
#include "SO3part.hpp"
#include "SO3vec.hpp"

using namespace cnine;
using namespace GElib;


int main(int argc, char** argv){
  cout<<endl;

  SO3type tau1({1,2,2});
  SO3type tau2({1,2,3});

  SO3vec u(tau1,fill::gaussian);
  SO3vec v(tau1,fill::gaussian);
  SO3part p(1,2,fill::gaussian);
  print("u",u);
  print("v",v);
  print("p",p);
  cout<<endl;

  CtensorPackObj W(dims(tau1,tau2),fill::gaussian);

  vector<SO3vec*> v_vec;

  //cout<<u[1]<<endl<<endl; 

  //u[1]=2*u[1]; 
  //u[1]=p;

  //cout<<u<<endl; 
  //u[2].fragment(0)=10*u[2].fragment(0);
  //cout<<u<<endl; 

  print("u",u);

  cout<<endl; 

  cout<<"---------"<<endl;

  //CscalarObj a=u(1,0,1);
  //complex<float> aval=u(1,0,1);
  //cout<<a<<endl;
  //cout<<aval<<endl;


}

  //for(int i=0; i<12; i++)
  //v_vec.push_back(new SO3vecObj(W*u));
  
  //cout<<*v_vec[7]<<endl;

    //SO3vecObj v=W*u;
    //cout<<v<<endl;
