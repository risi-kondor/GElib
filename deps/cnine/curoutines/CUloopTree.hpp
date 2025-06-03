#ifndef _CUloopTree
#define _CUloopTree

namespace cnine{

  class CUloopTree{
  public:
    
    typedef ctree_index_set IXSET;

    shared_ptr<LoopTreeNode> root;
    shared_ptr<LoopTree_tensor_registry> registry;


    CUloopTree(){
      registry=to_share(new LoopTree_tensor_registry());
      root=to_share(new LoopTreeNode(registry));
    }
    
    pair<CUloopTree,shared_ptr<LoopTreeNode> > marked_copy(const shared_ptr<LoopTreeNode>& marked){
      CUloopTree R;
      R.registry=to_share(new LoopTree_tensor_registry());
      R.root=to_share(new LoopTreeNode(R.registry));
      return make_pair(R,R.root->copy_subtrees(src->root,marked);
    }

  };

}

#endif 
