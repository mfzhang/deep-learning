/*************************************************************************
    > File Name: preprocess.cpp
    > Author: chenrudan
    > Mail: chenrudan123@gmail.com
    > Created Time: 2014年07月14日 星期一 10时57分40秒
 ************************************************************************/

#include <iostream>
#include <cmath>
#include "preprocess.h"

using namespace std;

Preprocess::Preprocess()
{
}

Preprocess::~Preprocess()
{
}

void Preprocess::BaseWhiten(Matrix *mat)
{
    double mean = 0;
    double std = 0;
    for(long i = 0; i < mat->GetRowNum(); i++)
    {
        for(long j = 0; j < mat->GetColNum(); j++)
        {
            mean += mat->GetElement(i, j);
        }
    }
    mean = mean/(mat->GetRowNum()*mat->GetColNum());
    for(long i = 0; i < mat->GetRowNum(); i++)
    {
        for(long j = 0; j < mat->GetColNum(); j++)
        {
            std += (mat->GetElement(i, j) - mean)*(mat->GetElement(i, j) - mean);
        }
    }
    std = sqrt(std/(mat->GetRowNum()*mat->GetColNum() - 1));
    for(long i = 0; i < mat->GetRowNum(); i++)
    {
        for(long j = 0; j < mat->GetColNum(); j++)
        {
            double value = (mat->GetElement(i, j) - mean)/std;
            mat->ChangeElement(i, j, value);
        }
    }
}






