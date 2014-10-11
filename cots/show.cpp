/*************************************************************************
    > File Name: show.cpp
    > Author: chenrudan
    > Mail: chenrudan123@gmail.com 
    > Created Time: 2014年08月27日 星期三 19时05分03秒
 ************************************************************************/

#include<iostream>
#include<fstream>
#include "/home/crd/crd/deeplearning/opencv/opencv-2.4.9/include/opencv/cv.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
using namespace std;
using namespace cv;

void show(string filename, int num, int channels, int size, string windowname);

int main()
{
	show("../data/layer1_0.bin", 40, 8, 10, "originwindow");
//	show("./data/R_picture_4.dat", 100, 3, 96, "8epoch");
//	show("./data/reconstruct.bin", 100, 3, 96, "4epoch");
//	show("./data/first_w.dat", 100, 8, 20, "4epoch");
	return 0;
}

void show(string filename, int num, int channels, int size, string windowname)
{
	ifstream fin(filename.c_str(), ios::binary);
	char *buffer = new char[num*size*size*channels];
	int pos = num*channels*size*size;
	fin.seekg(pos, fin.beg);
	fin.read(buffer, num*size*size*channels);
	float *input = new float[num*size*size*channels];
	for(int i = 0; i < num; i++)
	{
	    for(int j = 0; j < channels; j++)
	    {
	        float sum = 0;
	        for(int m = 0; m < size; m++)
	        {
	            for(int n = 0; n < size; n++)
	            {
	              //  sum += buffer[i*channels*size*size + j*size*size + m*size + n];
	                unsigned char tmp = buffer[i*channels*size*size + j*size*size + m*size + n];
	                sum += tmp;
	                cout << (int)buffer[i*channels*size*size + j*size*size + m*size + n] << "\t";
	            }
	        }
	        float average = sum/(size*size);
	        for(int m = 0; m < size; m++)
	        {
	            for(int n = 0; n < size; n++)
	            {
	                input[i*channels*size*size + j*size*size + m*size + n] = buffer[i*channels*size*size + j*size*size + m*size + n] - average;
	            }
	        }
	        float square = 0;
	        for(int m = 0; m < size; m++)
	        {
	            for(int n = 0; n < size; n++)
	            {
	                square += input[i*channels*size*size + j*size*size + m*size + n]*input[i*channels*size*size + j*size*size + m*size + n];
	            }
	        }
	        for(int m = 0; m < size; m++)
	        {
	            for(int n = 0; n < size; n++)
	            {
	                input[i*channels*size*size + j*size*size + m*size + n] = input[i*channels*size*size + j*size*size + m*size + n]/sqrt(square);
	                cout << input[i*channels*size*size + j*size*size + m*size + n] << "\t";
	            }
	        }

	    }
	}
	delete[] buffer;
    
}











