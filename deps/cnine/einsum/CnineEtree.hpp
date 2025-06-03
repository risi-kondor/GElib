#ifndef _Etree
#define _Etree

#include "EtreeNode.hpp"
#include "EtreeLoopNode.hpp"
#include "EtreeContractionNode.hpp"
#include "EtreeArgNode.hpp"
#include "EtreeParams.hpp"
#include "ContractionTree.hpp"


namespace cnine{


  class Etree{
  public:

    shared_ptr<EtreeNode> root;
    vector<int> loop_order;

    Etree(){}

    Etree(const ContractionTree& x, vector<int> _loop_order):
      loop_order(_loop_order){
      root=make_shared<EtreeLoopNode>(-1);
      translate_subtree(*x.root);
    }

    void insert(const shared_ptr<EtreeNode> x){
      auto node=root;
      auto external=x->external_indices.order(loop_order);
      for(auto ix:external){
	//for(int i=0; i<loop_order.size(); i++)
	//if(x->external_indices.find(loop_order[i])){
	auto p=node->subtree(ix);
	if(p) node=p;
	else{
	  auto new_node=make_shared<EtreeLoopNode>(ix);
	  node->subtrees.push_back(new_node);
	  node=new_node;
	}
      }
      node->children.push_back(x);
    }

    void translate_subtree(const ContractionNode& x){
      for(auto& p:x.children)
	translate_subtree(*p);
      if(x.arg_id>=0){
	//insert(new EtreeArgNode(x.arg_id,x.indices));
      }
      if(x.arg_id==-1){
	insert(to_share(new EtreeContractionNode(x.contraction_index,x.external_indices)));
      }
    }

    //void insert_node(shared_ptr<EtreeNode> x){
    //vector<int> ordered_indices;
    //for(auto& p:x.indices)
    //}


  public:

    //void translate_depth_first(const Contraction

    void insert_node(const shared_ptr<EtreeNode>& parent, const shared_ptr<EtreeNode>& node){
      //if(node->rtensor==0) node->rtensor=parent->rtensor;
      parent->children.push_back(node);
    }

    shared_ptr<EtreeNode> insert_loop(const shared_ptr<EtreeNode>& parent, const int ix){
      shared_ptr<EtreeNode> node=to_share(new EtreeLoopNode(ix));
      insert_node(parent,node);
      return node;
    }

    shared_ptr<EtreeNode> insert_contraction(const shared_ptr<EtreeNode>& parent, const int ix){
      shared_ptr<EtreeNode> node=to_share(new EtreeContractionNode(ix));
      insert_node(parent,node);
      return node;
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------

    


    string cpu_code(const EtreeParams& params){
      ostringstream oss;
      oss<<"#include<TensorView.hpp>\n";
      oss<<"using namespace cnine;\n";
      oss<<"\n";
      oss<<"void einsum(const TensorView<float>& r, vector<const TensorView<float> >& args){\n\n";
      oss<<root->cpu_code(params,1);
      oss<<"}";
      return oss.str();
    }


    string str(const string indent="") const{
      return root->str();
    }

    friend ostream& operator<<(ostream& stream, const Etree& x){
      stream<<x.str(); return stream;
    }



  };


}
#endif 

