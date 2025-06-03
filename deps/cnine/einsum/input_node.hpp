#ifndef _input_node
#define _input_node

#include "einsum_node.hpp"


namespace cnine{

  class input_node: public einsum_node{
  public:
    
    vector<vector<int> > map_to_dims;

    input_node(const string _name, vector<vector<int> >& _map_to_dims):
      einsum_node(_map.size(),_name),
      map_to_dims(_map_to_dims){}

  };
}

#endif 
