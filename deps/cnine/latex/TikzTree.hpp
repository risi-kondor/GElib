#ifndef _CnineTikzTree
#define _CnineTikzTree

#include "TikzStream.hpp"
#include "TikzTreeNode.hpp"


namespace cnine{


  class TikzTree{
  public:

    shared_ptr<TikzTreeNodeObj> root;

    TikzTree(string label="", string rlabel=""){
      //root=make_shared<TikzTreeNodeObj>(label,rlabel);
    }

    TikzTreeNode add_root(string label="",string rlabel=""){
      return root=make_shared<TikzTreeNodeObj>(label,rlabel);
      //return root->add_child(label,rlabel);
    }

    string latex() const{
      TikzStream tstream;
      tstream<<"\\begin{tikzpicture}[";
      tstream<<"default/.style={circle,draw=black}";
      //tstream<<"doublec/.style={double circle,draw=black}";
      //tstream<<",main node/.style={circle, draw=black}";
      //tstream<<",child node/.style={circle, draw=black}";
      tstream<<",sibling distance=3cm";
      tstream<<"]\n";
      if(root){
	tstream.oss<<"\\";
	root->write_latex(tstream);
	tstream.oss<<";\n";
      }
      tstream.write("\\end{tikzpicture}");
      return tstream.str();
    }

  };

}


#endif 
