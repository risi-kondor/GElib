#ifndef _CUloopTreeNodes
#define _CUloopTreeNodes

namespace cnine{

  class KernelLoopTreeNode: public LoopTreeNode{
  public:

    KernelLoopTreeNode(const LoopTreeNode& x):
      LoopTreeNode(x){}
  };

}

#endif 
