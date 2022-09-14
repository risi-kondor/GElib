#ifndef _SO3CG_addCGproductPrg
#define _SO3CG_addCGproductPrg

#include "SO3CGprogram.hpp"
#include "SO3CGP_addCGproductOp.hpp"


namespace GElib{


  class SO3CGP_addCGproductPrg: public SO3CGprogram{
  public:

    SO3CGP_addCGproductPrg(const SO3CGproductSignature& taus){
      output_nodes=recurse(0,taus);
    }
 

  private:


    vector<int> recurse(int start, const SO3CGproductSignature& signature){

      if(signature.size()==1){
	const SO3type& tau=signature[0];
	const int L=tau.getL();
	vector<int> R(L+1);
	for(int l=0; l<=L; l++)
	  if(tau[l]>0) R[l]=add_input_node(l,tau[l],start);
	  else R[l]=-1;
	return R;
      }

      int split=signature.size()/2;
      vector<int> xnodes=recurse(start,
	SO3CGproductSignature(vector<SO3type>(signature.begin(),signature.begin()+split),signature.maxl));
      vector<int> ynodes=recurse(start+split,
	SO3CGproductSignature(vector<SO3type>(signature.begin()+split,signature.end()),signature.maxl));

      vector<int> R(xnodes.size()+ynodes.size()-1,-1);

      for(int l1=0; l1<xnodes.size(); l1++){
	if(xnodes[l1]==-1) continue;
	SO3CGnode& xnode=*nodes[xnodes[l1]];

	for(int l2=0; l2<ynodes.size(); l2++){
	  if(ynodes[l2]==-1) continue;
	  SO3CGnode& ynode=*nodes[ynodes[l2]];
	  
	  for(int l=std::abs(l1-l2); l<=l1+l2; l++){
	    if(signature.maxl>-1 && l>signature.maxl) break;

	    if(R[l]==-1) R[l]=add_node(l,0);
	    SO3CGnode& rnode=*nodes[R[l]];
	    rnode.add_op(new SO3CGP_addCGproductOp(rnode,xnode,ynode,rnode.n));
	    rnode.n+=xnode.n*ynode.n;
	  }
	}
      }
      return R;
    }


    
  };


  /*
  inline SO3CGprogram::Node* CGproduct(const SO3CGprogram::Node& x, const SO3CGprogram::Node& y){
    assert(x.owner==y.owner);
    SO3CGprogram::Node* node=new SO3CGprogram::Node(x.owner,CGproduct(x.tau,y.tau));
    node->args.push_back(const_cast<SO3CGprogram::Node*>(&x));
    node->args.push_back(const_cast<SO3CGprogram::Node*>(&y));
    //node->args.push_back(&y);
    //x.owner->nodes.push_back(node);
    return node;
  }
  */

}


#endif


