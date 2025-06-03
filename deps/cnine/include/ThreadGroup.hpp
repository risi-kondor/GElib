
/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2021, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */


#ifndef _ThreadGroup
#define _ThreadGroup

#include <thread>
#include <condition_variable>
#include <atomic>

namespace cnine{

  extern thread_local int nthreads;


  class ThreadGroup{
  public:

    int maxthreads=4;
    mutex mx;
    condition_variable start_thread_cv;
    mutex start_thread_mx;
    atomic<int> nrunning;
    vector<thread> threads;


  public:


    ThreadGroup()=delete;
  
    ThreadGroup(const int _maxthreads=4): 
      maxthreads(_maxthreads){
      nrunning=0;
    }

    ~ThreadGroup(){
      for(auto& p:threads)
	p.join();
    }


  public:

    template<typename FUNCTION, typename OBJ>
    void add(const int nsubthreads, FUNCTION lambda, OBJ arg0){
      lock_guard<mutex> guard(mx);

      if(nrunning>=maxthreads){
	unique_lock<mutex> lock(start_thread_mx);
	start_thread_cv.wait(lock,[this](){return nrunning<maxthreads;});
      }

      nrunning++;
      threads.push_back(thread([this,nsubthreads,lambda,arg0](){
	    nthreads=nsubthreads;
	    lambda(arg0);
	    this->done();
	  }));
    }	 


    void done(){
      nrunning--;
      start_thread_cv.notify_one();
    }

  };


}

#endif 
