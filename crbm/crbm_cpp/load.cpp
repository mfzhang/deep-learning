/*************************************************************************
    > File Name: load_data.cpp
    > Author: chenrudan
    > Mail: chenrudan123@gmail.com
    > Created Time: 2014年07月17日 星期四 09时02分33秒
 ************************************************************************/

#include <iostream>
#include <fstream>
#include <vector>
#include "load.h"

using namespace std;

Load::Load()
{

}

Load::~Load()
{

}


vector<float>* Load::LoadData(string filename)
{
    ifstream fin(filename.c_str(), ios::binary);
    unsigned char buffer;
    vector<float> *data = new vector<float>;
    while(!fin.eof())
    {
        fin >> buffer;
        data->push_back(buffer);
    }
    fin.close();
    return data;
}
