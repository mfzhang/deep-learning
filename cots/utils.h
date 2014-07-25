/*************************************************************************
    > File Name: utils.h
    > Author: chenrudan
    > Mail: chenrudan123@gmail.com
    > Created Time: 2014年07月15日 星期二 09时23分18秒
 ************************************************************************/
#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <cstdlib>
#include <cmath>

using namespace std;


inline float RandomWeight(float low, float upper){
    return (rand() * 1.0 / RAND_MAX) * (upper - low) + low;
}

inline float RandomNumber(){
    return (rand() / (RAND_MAX + 1.0));
}

inline bool CompareFloat(float a, float b){
    return (a > b);
}

inline float Logisitc(float a){
    return 1.0 / (1 + exp(-a));
}



#endif
