/*************************************************************************
    > File Name: test.cpp
    > Author: chenrudan
    > Mail: chenrudan123@gmail.com
    > Created Time: 2014年07月13日 星期日 17时09分45秒
 ************************************************************************/

#include <iostream>
#include "matrix.h"
#include "preprocess.h"
#include "conv.h"

using namespace std;

int main()
{
	Matrix mat_1(4,4),mat_2(2,2), *mat_3;
	for(int i = 0; i < 4; i++)
	{
		for(int j = 0; j < 4; j++)
			mat_1.AddElement(j);
	}
	for(int i = 0; i < 2; i++)
	{
		for(int j = 0; j < 2; j++)
			mat_2.AddElement(j);
	}

	//mat_3 = Matrix::MatrixMultiply(&mat_1, &mat_2);
	//Preprocess::BaseWhiten(mat_3);
	mat_3 = Conv::Conv2d(&mat_1, &mat_2, 2);
	for(int i = 0; i < 2; i++)
	{
		for(int j = 0; j < 2; j++)
			cout << mat_3->GetElement(i,j) << endl;
	}

	delete mat_3;
	return 0;

}
