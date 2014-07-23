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


float* Load::LoadData(string filename)
{
    ifstream fin(filename.c_str(), ios::binary|ios::ate);
    char *buffer;
    float *data;
    fin.seekg(0, fin.end);
    long m = fin.tellg();
    buffer = new char[m];
    data = new float[m];
    fin.seekg (0, fin.beg);
    fin.read(buffer, m);
    for(int i = 0; i < m; i++)
    {
        unsigned char u_buffer = buffer[i];
        data[i] = u_buffer;
    }
    fin.close();
    delete buffer;
    return data;
}

float* LoadPartData(string filename, long start, long end)
{
    ifstream fin(filename.c_str(), ios::binary|ios::ate);
    char *buffer;
    float *data;
    long length = end -start;
    buffer = new char[length];
    data = new float[length];
    fin.seekg(start, fin.beg);
    fin.read(buffer, length);
    for(int i = 0; i < length; i++)
    {
        unsigned char u_buffer = buffer[i];
        data[i] = u_buffer;
    }
    fin.close();
    delete buffer;
    return data;
}
