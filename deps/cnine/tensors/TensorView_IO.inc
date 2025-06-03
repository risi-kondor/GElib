/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2021--, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */


static string classname(){
  return "TensorView<"+dtype_str()+">";
}

static string dtype_str(){
  if constexpr(std::is_same<TYPE,int>::value) return "int";
  if constexpr(std::is_same<TYPE,float>::value) return "float";
  if constexpr(std::is_same<TYPE,double>::value) return "double";
  if constexpr(std::is_same<TYPE,complex<float> >::value) return "complex<float>";
  if constexpr(std::is_same<TYPE,complex<double> >::value) return "complex<double>";
  return "unknown";
}

string describe() const{
  ostringstream oss;
  oss<<"TensorView<"<<dtype_str()<<">"<<dims<<" ["<<strides<<"]"<<endl;
  return oss.str();
}

string repr() const{
  ostringstream oss;
  oss<<"TensorView<"<<dtype_str()<<">"<<dims<<" [strides="<<strides<<"]";
  return oss.str();
}

string to_print(const string indent="") const{
  ostringstream oss;
  oss<<indent<<repr()<<":"<<endl;
  oss<<str(indent+"  ")<<endl;
  return oss.str();
}

string to_string(const string indent="") const{
  if(dev>0) return TensorView(*this,0).str(indent);
  ostringstream oss;

  //TYPE largest=std::max(-min(),max());
  float limit;
  if constexpr(std::is_same<TYPE,complex<float> >::value)
    limit=std::real(max_abs())/10e5;
  else 
    limit=max_abs()/10e5;

  if(ndims()==1){
    oss<<base_indent<<indent<<"[ ";
    for(int i0=0; i0<dims[0]; i0++)
      if(abs((*this)(i0))>limit) oss<<(*this)(i0)<<" ";
      else oss<<TYPE(0)<<" ";
    oss<<"]"<<endl;
    return oss.str();
  }

  if(ndims()==2){
    for(int i0=0; i0<dims[0]; i0++){
      oss<<base_indent<<indent<<"[ ";
      for(int i1=0; i1<dims[1]; i1++)
	if(abs((*this)(i0,i1))>limit) oss<<(*this)(i0,i1)<<" ";
	else oss<<TYPE(0)<<" ";
      oss<<"]"<<endl;
    }
    return oss.str();
  }

  if(ndims()>2){
    Gdims adims=dims.chunk(0,ndims()-2);
    adims.for_each_index([&](const Gindex& ix){
	oss<<base_indent<<indent<<"Slice"<<ix<<":"<<endl;
	oss<<slice(ix).to_string(base_indent+indent+"  ")<<endl;
      });
  }

  return oss.str();
}


string str(const string indent="") const{
  ostringstream oss;
  oss<<to_string(indent);
  return oss.str();
}


friend ostream& operator<<(ostream& stream, const TensorView<TYPE>& x){
  stream<<x.str(); return stream;
}
