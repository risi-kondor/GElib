#ifndef _MakeCUloopTrees
#define _MakeCUloopTrees

namspace cnine{

  class MakeCUloopTrees{
  public:

    MakeCUloopTrees(const LoopTree& ltree){
      CUloopTree ltree;
      
    }

    spawn_recursively(const CUloopTree& tree, const LoopTreeNode& src, const LoopTreeNode& cpy){
      for(auto& child:src.children){
	if(child->is_parallelizable()){
	  CUloopTree new_tree(tree);
	}
      }
    }
    
    copy

  };

}

#endif 
