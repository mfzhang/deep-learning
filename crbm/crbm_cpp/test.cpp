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
#include "matrix.h"
#include "preprocess.h"
#include "conv.h"
#include "crbm.h"

using namespace std;

int main()
{
    Crbm layer_1;
    vector<Matrix*> *input_image = new vector<Matrix*>;
    vector<Matrix*> *output_image = new vector<Matrix*>;
    ifstream fin("./data_batch_1.bin", ios::binary);
    char buffer[3073];
    fin.read(buffer, 3073);
    int k = 1;
    for(int j = 0; j < 3; j++)
    {
        Matrix *new_mat = new Matrix(32, 32);
        for(int m = 0; m < 32; m++)
        {
            for(int n = 0; n < 32; n++)
            {
                new_mat->AddElement(atoi(&buffer[k]));
                k++;
                cout << new_mat->GetElement(m,n) << "\n";
            }
            cout << "-------------------------------";
        }
        Preprocess::BaseWhiten(new_mat);
        input_image->push_back(new_mat);
    }
    fin.close();
    output_image = layer_1.RunBatch(5, 10, 3, 32, input_image);

	return 0;

}
