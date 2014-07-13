/*************************************************************************
    > File Name: test.cpp
    > Author: chenrudan
    > Mail: chenrudan123@gmail.com
    > Created Time: 2014年07月13日 星期日 17时09分45秒
 ************************************************************************/

#include<iostream>
#include"matrix.h"

using namespace std;

int main()
{
	Matrix mat_1(3,2),mat_2(2,3), mat_3(3,3);
	for(int i = 0; i < 3; i++)
	{
		for(int j = 0; j < 2; j++)
			mat_1.AddElement(1);
	}
	for(int i = 0; i < 2; i++)
	{
		for(int j = 0; j < 3; j++)
			mat_2.AddElement(1);
	}

	mat_3 = mat_3.MatrixMultiply(mat_1, mat_2);
	for(int i = 0; i < 3; i++)
	{
		for(int j = 0; j < 3; j++)
			cout << mat_3.GetElement(i,j) << endl;
	}
	return 0;

}
