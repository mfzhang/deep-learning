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
    ifstream fin(filename.c_str(), ios::binary|ios::ate);
    char *buffer;
    fin.seekg(0, fin.end);
    long m = fin.tellg();
    buffer = new char[m];
    fin.seekg (0, fin.beg);
    vector<float> *data = new vector<float>;
    fin.read(buffer, m);
    for(int i = 0; i < m; i++)
    {
        unsigned char u_buffer = buffer[i];
        data->push_back((float)u_buffer);
    }
    fin.close();
    return data;
}
