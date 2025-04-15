#include "GElib_base.cpp"
#include "O3element.hpp"

using namespace cnine;
using namespace GElib;


int main(int argc, char** argv){
  //GElib::GElibSession session;
  cout<<endl;

  O3element<float> R0=O3element<float>::identity();
  cout<<R0<<endl;

  O3element<float> R1=O3element<float>::random();
  cout<<R1<<endl;
  cout<<R1*R1<<endl;
  cout<<R1.inv()*R1<<endl;

}
