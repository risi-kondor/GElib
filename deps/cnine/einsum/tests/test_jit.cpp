/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2023, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */


#include "Cnine_base.cpp"
#include "CnineSession.hpp"
#include "Einsum2.hpp"
#include <dlfcn.h>
#include <cstdlib>
#include <fstream>

typedef double (*func_ptr)(double);

using namespace cnine;


std::string generate_code(int param) {
    std::stringstream ss;
    ss << "#include <cmath>\n"
       << "extern \"C\" double my_function(double x) {\n"
       << "    return std::pow(x, " << param << ");\n"
       << "}\n";
    return ss.str();
}

void write_code_to_file(const std::string& code, const std::string& filename) {
  std::ofstream file(filename);
  file << code;
    file.close();
}

void compile_code(const std::string& filename) {
  std::string command = "clang -shared -fPIC " + filename + " -o libdynamic.so" +"> output.txt";
    system(command.c_str());
}

func_ptr load_function() {
  void* handle = dlopen("./libdynamic.so", RTLD_LAZY);
  if (!handle) {
    throw std::runtime_error(dlerror());
  }
  func_ptr func = (func_ptr)dlsym(handle, "my_function");
  if (!func) {
    throw std::runtime_error(dlerror());
  }
  return func;
}

int main(int argc, char** argv){

  int param = 3;
  std::string code = generate_code(param);
  write_code_to_file(code, "dynamic_code.dcpp");
  compile_code("dynamic_code.dcpp");
  
  func_ptr dynamic_func = load_function();
  double result = dynamic_func(2.0);
  std::cout << "Result: " << result << std::endl;
    
}

