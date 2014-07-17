/*************************************************************************
  > File Name: test.cpp
  > Author: chenrudan
  > Mail: chenrudan123@gmail.com
  > Created Time: 2014年07月13日 星期日 17时09分45秒
 ************************************************************************/

#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "matrix.h"
#include "preprocess.h"
#include "conv.h"
#include "crbm.h"
#include "load.h"

using namespace std;
using namespace cv;

int main()
{
	Crbm layer_1;
	vector<Matrix*> input_image;
	vector<Matrix*> output_image;
	Load load;
	//参数初值
	int batch_size = 2;
	int filter_size = 5;
	int image_size = 32;
	int pooling_size = 2;
	int input_channels = 3;
	int lay1_channels = 10;


	vector<float> *file_data = load.LoadData("./data_batch_1.bin");
	//给100个图片赋初值
	int k = 1;
	for(int i = 0; i < 100; i++)
	{
		Matrix *p_new_mat = new Matrix[3];
		for(int j = 0; j < 3; j++)
		{
			p_new_mat[j].init(32,32);
			for(int m = 0; m < 32; m++)
			{
				for(int n = 0; n < 32; n++)
				{
					p_new_mat[j].AddElement(file_data->at(k));
					k++;
				}
			}
			Preprocess::BaseWhiten(p_new_mat + j);
		}
		k++;
		input_image.push_back(p_new_mat);
	}
	int pos =0;
	//1.初始化参数
	layer_1.FilterInit(filter_size, lay1_channels, input_channels, image_size, batch_size, pooling_size);
	for(int batch = 0; batch < 10/batch_size; batch++)
	{
		output_image = layer_1.RunBatch(input_image, pos);
		pos += 2;
	}

	//显示权重
	vector<Matrix*> weight = layer_1.GetWeight();
	int w_size = weight.size();
	for(int i = 0; i < w_size; i++)
	{
		cout << i << endl;
		cout << "-----------------------\n";
		for(int m = 0; m < 5; m++)
		{
			for(int n = 0; n < 5; n++)
			{
				cout << weight[i]->GetElement(m, n) << "\n";
			}
		}
	}

	//显示第一层输出
	ofstream layer1_output("layer1.txt");
	int o_size = output_image.size();
	for(int i = 0; i < o_size; i++)
	{
		cout << i << endl;
		cout << "-----------------------\n";
		for(int m = 0; m < 28; m++)
		{
			for(int n = 0; n < 28; n++)
			{
			//	cout << output_image[i]->GetElement(m, n) << "\n";
				layer1_output << output_image[i]->GetElement(m, n) << "\n";
			}
		}
	}
	layer1_output.close();

	vector<Mat> cv_layer1;
	for(int i = 0; i < o_size; i++)
	{   
		Mat tmp(28,28,CV_32F);
		cout << "-----------------------\n";
		for(int m = 0; m < 28; m++)
		{   
			for(int n = 0; n < 28; n++)
			{   
				tmp.at<float>(m,n) = output_image[i]->GetElement(m, n);
		//		cout << tmp.at<float>(m,n) << "\n";
			}   
		}   
		cv_layer1.push_back(tmp);
		namedWindow("OutputImage");
		imshow("OutputImage", tmp); 
	}
	

	return 0;
}
