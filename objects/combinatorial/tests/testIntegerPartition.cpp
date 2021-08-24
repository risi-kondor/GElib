#include "GElib_base.cpp"
#include "GElibSession.hpp"
#include "CombinatorialClasses.hpp"

using namespace cnine;
using namespace GElib;


int main(int argc, char** argv){
  GElibSession session;
  CombinatorialClasses combi_classes;
  cout<<endl;

  IntegerPartition lambda({4,3,1});
  cout<<lambda<<endl<<endl;

  IntegerPartitions IP(5);
  for(int i=0; i<IP.size(); i++)
    cout<<IP[i]<<endl;


}

