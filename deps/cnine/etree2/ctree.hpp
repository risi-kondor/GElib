#ifndef _ctree
#define _ctree

#include "ctree_index_set.hpp"


namespace cnine{

  class ctree;

  typedef ctree_index_set IXSET;


  class ctree_tensor_node{
  public:

    ctree* owner;
    int id;
    IXSET indices;
    vector<int> dependents;

    ctree_tensor_node(ctree* _owner, const int _id, const IXSET& _indices):
      owner(_owner), id(_id), indices(_indices){}

    string element_str() const{
      ostringstream oss;
      oss<<"T"<<id<<"("<<indices.limit_str()<<")";
      return oss.str();
    }

    virtual string str() const{
      ostringstream oss;
      oss<<"T"<<id<<"=Tensor("<<indices.limit_str()<<")";
      return oss.str();
    }

  };


  class ctree_contraction_node: public ctree_tensor_node{
  public:

    int ix;
    vector<shared_ptr<ctree_tensor_node> > args;

    ctree_contraction_node(ctree* _owner, const int _id, const shared_ptr<ctree_tensor_node>& x, 
      const shared_ptr<ctree_tensor_node>& y, const int _ix):
      ctree_tensor_node(_owner,_id,x->indices.contract(y->indices,_ix)), 
      ix(_ix){
      args.push_back(x);
      args.push_back(y);
      x->dependents.push_back(id);
      y->dependents.push_back(id);
    }

    string str() const{
      ostringstream oss;
      oss<<"T"<<id<<"=sum_{"<<ix<<"}";
      for(auto& p:args)
	oss<<p->element_str()<<"*";
      oss<<"\b";
      return oss.str();
    }
    
  };


  class ctree_tensor_handle{
  public:

    shared_ptr<ctree_tensor_node> obj;

    ctree_tensor_handle(const shared_ptr<ctree_tensor_node>& _obj): 
      obj(_obj){}



  };



  class ctree{
  public:

    vector<shared_ptr<ctree_tensor_node> > nodes;

    ctree(){}

    ctree(const int i){}


  public:

    ctree_tensor_handle add_input(const IXSET& _indices){
      auto r=make_shared<ctree_tensor_node>(this,nodes.size(),_indices);
      nodes.push_back(r);
      return r;
    }

    string str() const{
      ostringstream oss;
      for(auto& p:nodes)
	oss<<p->str()<<endl;
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const ctree& x){
      stream<<x.str(); return stream;
    }

  };


  ctree_tensor_handle contract(const ctree_tensor_handle& x, const ctree_tensor_handle& y, const int ix){
    auto r=make_shared<ctree_contraction_node>(x.obj->owner,x.obj->owner->nodes.size(),x.obj,y.obj,ix);
    x.obj->owner->nodes.push_back(r);
    return shared_ptr<ctree_tensor_node>(r);
    //return r;
  }

}

#endif 
