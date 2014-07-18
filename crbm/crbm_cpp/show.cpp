/*************************************************************************
    > File Name: show.cpp
    > Author: chenrudan
    > Mail: chenrudan123@gmail.com
    > Created Time: 2014年07月18日 星期五 09时22分33秒
 ************************************************************************/

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "show.h"
#include "matrix.h"

using namespace std;
using namespace cv;

Show::Show()
{

}

Show::~Show()
{

}

void Show::ShowMyMatrix(Matrix* m)
{
    int row = m->GetRowNum();
    Mat tmp(row, row, CV_32F);
    for(int i = 0; i < row; i++)
    {
        for(int j = 0; j < row; j++)
        {
            tmp.at<float>(i,j) = m->GetElement(i, j);
        }
    }
    namedWindow("OutputImage", WINDOW_AUTOSIZE);
    imshow("OutputImage", tmp);
    waitKey();
}













