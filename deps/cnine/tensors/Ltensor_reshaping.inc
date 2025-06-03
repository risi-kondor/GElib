

/*
Ltensor with_decreasing_strides(){
  auto p=strides.decreasing_ordering();
  return Ltensor(arr,dims.permute(p),strides.permute(p));
}

Ltensor scrunch(){
  int n=ndims();
  Gdims _dims;
  GstridesB _strides;
  auto ordering=strides.ordering();
  for(int i=0; i<n; i++){
    int s=strides[ordering[i]];
    int d=dims[ordering[i]];
    while(i+1<n && strides[ordering[i+1]]==s*d){
      d*=dims[ordering[i+1]];
      i++;
    }
    _dims.push_back(d);
    _strides.push_back(s);
  }
  return Ltensor(get_arr(),std::reverse(_dims.begin(),_dims.end()),std::reverse(_strides.begin(),strides.end()));
}


*/
