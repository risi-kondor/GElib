
#include "Cnine_base.cpp"
#include "CnineSession.hpp"
#include "sparse_graph.hpp"
#include "FindPlantedSubgraphs.hpp"
#include "FindPlantedSubgraphs2.hpp"

using namespace cnine;

typedef sparse_graph<int,float> Graph;


int main(int argc, char** argv){

  cnine_session session;

  Graph G=Graph::random(12,0.3);
  int_pool Gd=G.as_int_pool();
  cout<<G<<endl;

  Graph H=Graph::cycle(5);
  int_pool Hd=H.as_int_pool();
  cout<<H<<endl;

  auto f=FindPlantedSubgraphs(G,H);
  cout<<Tensor<int>(f)<<endl;

  auto fd=FindPlantedSubgraphs2(Gd,Hd);
  cout<<fd.matches<<endl;
  
}
