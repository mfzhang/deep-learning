/*************************************************************************
    > File Name: showweight.cpp
    > Author: chenrudan
    > Mail: chenrudan123@gmail.com 
    > Created Time: 2014年09月24日 星期三 20时46分58秒
 ************************************************************************/

#include<iostream>
#include<iostream>
#include<fstream>
#include "/home/crd/crd/deeplearning/opencv/opencv-2.4.9/include/opencv/cv.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

void scaleToUnitInterval(Mat &dispImg, ifstream &fin, int size, int channels, int pos);

int main()
{
	ifstream fin1("../data/mnist_layer1_w.bin", ios::binary);
	ifstream fin2("../data/mnist_layer1_5.bin", ios::binary);
	ifstream fin3("../data/mnist_layer1_6.bin", ios::binary);
	int num = 6;
	int size = 10;
	int channels = 20;
	for(int i = 0; i < num; i++)
	{
		Mat dispImg;
		
		dispImg.create(Size((size + 1)*channels, 3*(1 + size)), CV_8U);
		
		scaleToUnitInterval(dispImg, fin1, size, channels, 0);
		scaleToUnitInterval(dispImg, fin2, size, channels, 1);
		scaleToUnitInterval(dispImg, fin3, size, channels, 2);

		namedWindow("three", WINDOW_NORMAL);
		imshow("three", dispImg);
		waitKey();
		cout << i << endl;
	}
	fin1.close();
	fin2.close();
	fin3.close();

	return 0;
}

void scaleToUnitInterval(Mat &dispImg, ifstream &fin, int size, int channels, int pos)
{
	for(int j = 0; j < channels; j++)
	{
		Mat tmp(size, size, CV_8U);
		float sum = 0;
		float max_value = -10000;
		float min_value = 10000;
		float sqrt_sum = 0;
		float *buffer = new float[size*size];
		fin.read((char *)buffer, sizeof(float)*size*size);
		
		
/*		for(int m = 0; m < size; m++)
		{
			for(int n = 0; n < size; n++)
			{
				mean += buffer(m*size + n)
				sqrt_sum += buffer[m*size + n]*buffer[m*size + n];
			}
		}*/
		
		for(int m = 0; m < size; m++)
		{
			for(int n = 0; n < size; n++)
			{
		//		buffer[m*size + n] = buffer[m*size + n]/sqrt(sqrt_sum);				
				buffer[m*size + n] = buffer[m*size + n];				
				if(max_value < buffer[m*size + n])
					max_value = buffer[m*size + n];
				if(min_value > buffer[m*size + n])
					min_value = buffer[m*size + n];
				cout << buffer[m*size + n] << "\t";	
	//			sum += buffer[m*size + n];		
			}
		}
	//	sum = sum/(size*size); 
		float dist = max_value - min_value;
		for(int m = 0; m < size; m++)
		{
			for(int n = 0; n < size; n++)
			{
				tmp.at<uchar>(n,m) = (buffer[m*size + n] - min_value)/dist*255;
			}
		}
		Mat imgROI = dispImg(Rect(j*(1+size), pos*size + 1, size, size));
    	resize(tmp, imgROI, Size(size, size));
    	delete[] buffer;	
	}
}


