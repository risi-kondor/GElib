
#include "Cnine_base.cpp"
#include "CnineSession.hpp"
#include "sparse_graph.hpp"
#include "FindPlantedSubgraphs.hpp"
#include "Ltensor.hpp"

using namespace cnine;

typedef sparse_graph<int,float> Graph;


int main(int argc, char** argv){

  cnine_session session;

  Graph triangle(3,{{0,1},{1,2},{2,0}});
  Graph square(4,{{0,1},{1,2},{2,3},{3,0}});

  cout<<triangle<<endl;

  Ltensor<int> deg(dims(3));
  deg.set(0,2);
  deg.set(1,3);
  deg.set(2,2);
  triangle.set_degrees(deg);

  cout<<triangle<<endl;

  //Graph G=Graph::random(5,0.5);
  Graph G(8);
  G.insert(triangle,{0,1,2});
  G.insert(triangle,{5,6,7});
  cout<<G<<endl;

  auto fn=FindPlantedSubgraphs(G,triangle);
  cout<<Tensor<int>(fn)<<endl;
  
  //cout<<CachedPlantedSubgraphs()(G,triangle)<<endl;
  //cout<<CachedPlantedSubgraphs()(G,triangle)<<endl;

}

