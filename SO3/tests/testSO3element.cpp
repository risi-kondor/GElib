#include "GElib_base.cpp"
#include "SO3element.hpp"

using namespace cnine;
using namespace GElib;


int main(int argc, char** argv){
  //GElib::GElibSession session;
  cout<<endl;

  SO3element<float> R0=SO3element<float>::identity();
  cout<<R0<<endl;

  SO3element<float> R1=SO3element<float>::random();
  cout<<R1<<endl;
  cout<<R1*R1<<endl;
  cout<<R1.inv()*R1<<endl;

}
