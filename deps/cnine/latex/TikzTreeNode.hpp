#ifndef _CnineTikzTreeNode
#define _CnineTikzTreeNode

#include "TikzStream.hpp" 


namespace cnine{

  class TikzTreeNodeObj{
  public:

    string label;
    string llabel;
    string rlabel;
    string style="default";

    vector<shared_ptr<TikzTreeNodeObj> > children;

    TikzTreeNodeObj(const string _label, const string _rlabel=""):
      label(_label),
      rlabel(_rlabel){}

    shared_ptr<TikzTreeNodeObj> add_child(string _label="", const string _rlabel=""){
      auto r=make_shared<TikzTreeNodeObj>(_label,_rlabel);
      children.push_back(r);
      return r;
    }
     
    void write_latex(TikzStream& tstream){
      string node_str="node";
      //if(llabel!=""||rlabel!=""){
	node_str+="["+style;
	if(llabel!=""||rlabel!="") node_str+=",";
	if(rlabel!="") node_str+="label={[yshift=-20pt,font=\\fontsize{6pt}{8pt}\\selectfont]right:"+rlabel+"}";
	if(rlabel!=""&&llabel!="") node_str+=", ";
	if(llabel!="") node_str+="label={[yshift=-0pt,font=\\fontsize{6pt}{8pt}\\selectfont]left:"+llabel+"}";
	node_str+="]";
	//}
      node_str+="{"+label+"}";
      tstream.write(node_str);
      //if(rlabel!="") tstream.write("node[label={[yshift=-20pt,font=\\fontsize{6pt}{8pt}\\selectfont]right:"+rlabel+"}]{"+label+"}");
      //else tstream.write("node{"+label+"}");
      //if(rlabel!="")
      //tstream.write("edge from parent node[right, text width=4cm, align=left] {"+rlabel+"}");
      for(auto& p:children){
	tstream.depth++;
	tstream.write("child{");
	p->write_latex(tstream);
	tstream.write("}");
	tstream.depth--;
      }
    }
 
  };


  class TikzTreeNode{
  public:

    shared_ptr<TikzTreeNodeObj> obj;

    TikzTreeNode(const shared_ptr<TikzTreeNodeObj> _obj):
      obj(_obj){}

    TikzTreeNode add_child(const string label="", const string rlabel=""){
      return obj->add_child(label,rlabel);
    }
    
  };

}

#endif 
