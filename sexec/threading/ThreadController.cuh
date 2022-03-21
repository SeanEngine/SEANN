//
// Created by DanielSun on 3/20/2022.
//

#ifndef SEANN_THREADCONTROLLER_CUH
#define SEANN_THREADCONTROLLER_CUH

#include "../../seblas/gemm/Tensor.cuh"
#include <thread>
#include "../../sexec/threading/ThreadController.cuh"

using namespace seblas;
using namespace std;
namespace sexec{

    //To use this, the first params of your function need to be the tid
    template<const int x, const int y, typename ...Args, typename ...Args0>
    void _alloc(void(*func)(Args...), Args0... args){
        vector<thread> threads;
        for(int i = 0; i < x; i++){
            for (int j = 0; j < y; j++){
                threads.push_back(thread(func, i, j, args...));
            }
        }

        for(auto& t : threads){
            t.join();
        }
    }

    template<const int x, typename ...Args, typename ...Args0>
    void _alloc(void(*func)(Args...), Args0... args){
        vector<thread> threads;
        for(int i = 0; i < x; i++){
            threads.push_back(thread(func, i, x, args...));
        }

        for(auto& t : threads){
            t.join();
        }
    }
}


#endif //SEANN_THREADCONTROLLER_CUH
