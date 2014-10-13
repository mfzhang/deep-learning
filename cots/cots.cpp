/*************************************************************************
  > File Name: cots.cpp
  > Author: chenrudan
  > Mail: chenrudan123@gmail.com 
  > Created Time: 2014年08月30日 星期六 20时26分30秒
 ************************************************************************/

#include<iostream>
#include <fstream>
#include <cmath>
#include <time.h>
#include <string>
#include <sstream>
#include <unistd.h>
#include "mpi.h"
#include "utils.h"
#include "cots.h"
extern "C"{
#include "cblas.h"
}

using namespace std;

Cots::Cots(int input_size, int input_channels, int filter_size, int filter_channels, int batch_size, int block_size, \
        int step, int process_num, int thread_num, int pooling_size, float learning_rate, float learning_rate_alpha, float alpha, float momentum, float lambda, string address)
{
    init(input_size, input_channels, filter_size, filter_channels, batch_size, block_size, step, process_num, \
            thread_num, pooling_size, learning_rate, alpha, learning_rate_alpha, momentum, lambda, address);
}

Cots::Cots()
{
}

Cots::~Cots()
{
}

void Cots::init(int input_size, int input_channels, int filter_size, int filter_channels, int batch_size, int block_size, \
        int step, int process_num, int thread_num, int pooling_size, float learning_rate, float learning_rate_alpha, float alpha, float momentum, float lambda, string address)
{
    this->_pars = new _Pars;
    this->_pars->filter_channels = filter_channels;
    this->_pars->filter_size = filter_size; 
    this->_pars->input_channels = input_channels;
    this->_pars->input_size = input_size; 
    this->_pars->batch_size = batch_size; 
    this->_pars->block_size = block_size;
    this->_pars->step = step;
    this->_pars->process_num = process_num;
    this->_pars->thread_num = thread_num;
    this->_pars->out_size = ((this->_pars->input_size - this->_pars->filter_size)/2 + 1)* 2;
    this->_pars->pooling_size = pooling_size;
    this->_pars->learning_rate = learning_rate;
    this->_pars->alpha = alpha;
    this->_pars->learning_rate_alpha = learning_rate_alpha;
    this->_pars->momentum = momentum;
    this->_address = address;
    this->_lambda = lambda;
    assignMemory();

}

void Cots::testModel(int me, int epoch, int batch_all_size, bool type)
{
    initWeight(me, false);

    for(int epoch_idx = 0; epoch_idx < epoch; epoch_idx++)
    {
        for(int batch_idx = 0; batch_idx < batch_all_size/this->_pars->batch_size; batch_idx++)
        {
            if(me == 0)
            {
                if(batch_idx == 0)
                {
                    cout << "=============================\n";
                    cout << "====epoch is " << epoch_idx<< "=====\n";
                }
                cout << "batch_idx is " << batch_idx << "\n";
                //预处理图片
                //                preprocess(this->_input_address, batch_idx, this->_pars->batch_size, this->_pars->input_channels, this->_pars->input_size);
            }
            filterLayer(me, batch_idx);
            poolingLayer(me, batch_idx);
            lcnLayer(me, batch_idx);
            if((me == 0) && (epoch_idx == epoch -1))
            {
                int length = this->_pars->out_size*this->_pars->out_size*this->_pars->filter_channels*this->_pars->batch_size;
                subSaveFile("./binaryfile/test_layer1out_unlabeled.bin", length, this->_pars->receive_lcn, false);
            }

        }
    }
}


void Cots::trainModel(int me, int epoch, int batch_all_size, bool type)
{
    int weight_length = this->_pars->process_num*this->_pars->filter_channels*this->_pars->block_size*this->_pars->block_size*this->_pars->filter_size*this->_pars->filter_size*this->_pars->input_channels;
    this->_pars->winc = new float[weight_length];

    initWeight(me, true);
    normalizeWeight();

	string savename;
    zeros(this->_pars->winc, weight_length);
    for(int epoch_idx = 0; epoch_idx < epoch; epoch_idx++)
    {
        for(int batch_idx = 0; batch_idx < batch_all_size/this->_pars->batch_size; batch_idx++)
        {
            int r_size = this->_pars->input_size*this->_pars->input_size*this->_pars->input_channels*this->_pars->batch_size;
            if(me == 0)
            {
                if(batch_idx == 0)
                {
                    cout << "=============================\n";
                    cout << "====epoch is " << epoch_idx<< "=====\n";
                }
                cout << "batch_idx is " << batch_idx << "\n";
                //预处理图片
                preprocess( batch_idx, this->_pars->batch_size, this->_pars->input_channels, this->_pars->input_size, this->_pars->input, type);
            }
            MPI_Bcast(this->_pars->input, r_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
            this->_delta_alpha = 0;

            filterLayer(me, batch_idx);
            poolingLayer(me, batch_idx);
            lcnLayer(me, batch_idx);
            updateW(me, batch_idx);
            normalizeWeight();
            if(me == 0)
            {
                updateAlpha();
                this->_pars->alpha += this->_pars->learning_rate_alpha*this->_delta_alpha;
            }
            MPI_Bcast(&this->_pars->alpha, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

            if(me ==0 )
            {
                cout << "==========input=============\n";
                for(int i = 0; i< 10; i++)
                {
                    cout << this->_pars->input[i] << "\t";
                }
                cout << endl;
                cout << "==========reconstruct=============\n";
                for(int i = 0; i< 10; i++)
                {
                    cout << this->_pars->receive_reconstruct[i]<< "\t";
                }
                cout << endl;
                cout << "==========hidden=============\n";
                for(int i = 0; i< 10; i++)
                {
                    cout<<this->_pars->receive_hidden[i] << "\t";
                }
                cout << endl;
                cout << "==========pooling=============\n";
                for(int i = 0; i< 10; i++)
                {
                    cout<< this->_pars->receive_pooling[i] << "\t";
                }
                cout << endl;
                cout << "==========lcn=============\n";
                for(int i = 0; i< 10; i++)
                {
                    cout<< this->_pars->receive_lcn[i]<< "\t";
                }
                cout << endl;  
                
                float cost = 0;
                for(int i = 0 ; i < this->_pars->input_size*this->_pars->input_size*this->_pars->input_channels*this->_pars->batch_size; i++)
                {
                    cost += (this->_pars->receive_reconstruct[i] - this->_pars->input[i])*(this->_pars->receive_reconstruct[i] - this->_pars->input[i]);
                }
                float cost1 = 0;
                for(int i = 0 ; i < this->_pars->out_size*this->_pars->out_size*this->_pars->filter_channels*this->_pars->batch_size; i++)
                {
                    cost1 += this->_pars->receive_pooling[i]*this->_lambda;
                }
                cout << "the delta alpha is " << this->_pars->learning_rate_alpha*this->_delta_alpha << "       "  << "the alpha is     " <<  this->_pars->alpha<< endl;            
                cout << "============================================================================================================\n";
                cout << "============================================================================================================\n";
                cout << "cost of reconstruct is  " << cost << "  cost of pooling is  " << cost1 <<"     total cost is " << cost + cost1 << endl;
                cout << "============================================================================================================\n";
                cout << "============================================================================================================\n";
				
                if(epoch_idx == epoch - 1)
				{
					int length = this->_pars->input_size*this->_pars->input_size*this->_pars->input_channels*this->_pars->batch_size;
					savename = this->_address + "reconstruct.bin";
					subSaveFile(savename, length, this->_pars->receive_reconstruct, true);
					length = this->_pars->out_size*this->_pars->out_size*this->_pars->filter_channels*this->_pars->batch_size;
					savename = this->_address + "hidden.bin";
					subSaveFile(savename, length, this->_pars->receive_hidden, false);
					savename = this->_address + "pooling.bin";
					subSaveFile(savename, length, this->_pars->receive_pooling, false);
					savename = this->_address + "out.bin";
					subSaveFile(savename, length, this->_pars->receive_lcn, false);
				}
            }
        }
    }
    delete[] this->_pars->winc;
}

void Cots::preprocess(int batch_idx, int num, int channels, int size, float *input, bool type)
{
	string filename = this->_address + "in.bin";
    ifstream fin(filename.c_str(), ios::binary);
    if(type)
    {
        char *buffer = new char[num*size*size*channels];
        int pos = batch_idx*num*channels*size*size;
        fin.seekg(pos, fin.beg);
        fin.read(buffer, num*size*size*channels);
        for(int i = 0; i < num; i++)
        {
            for(int j = 0; j < channels; j++)
            {
                float sum = 0;
                for(int m = 0; m < size; m++)
                {
                    for(int n = 0; n < size; n++)
                    {
                        //		                sum += buffer[i*channels*size*size + j*size*size + m*size + n];
                        unsigned char tmp = buffer[i*channels*size*size + j*size*size + m*size + n];
                        sum += tmp;
                    }
                }
                float average = sum/(size*size);
                for(int m = 0; m < size; m++)
                {
                    for(int n = 0; n < size; n++)
                    {
                        unsigned char tmp = buffer[i*channels*size*size + j*size*size + m*size + n];
                        input[i*channels*size*size + j*size*size + m*size + n] = tmp - average;
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
                    }
                }

            }
        }
        delete[] buffer;
    }
    else
    {
        float *buffer = new float[num*size*size*channels];
        int pos = batch_idx*num*channels*size*size*sizeof(float);
        fin.seekg(pos, fin.beg);
        fin.read((char *)buffer, num*size*size*channels*sizeof(float));
        for(int i = 0; i < num; i++)
        {
            for(int j = 0; j < channels; j++)
            {
                
                float sum = 0;
                for(int m = 0; m < size; m++)
                {
                    for(int n = 0; n < size; n++)
                    {
                        sum += buffer[i*channels*size*size + j*size*size + m*size + n];
                    }
                }
                float average = (float)sum/(size*size);
                for(int m = 0; m < size; m++)
                {
                    for(int n = 0; n < size; n++)
                    {
                        input[i*channels*size*size + j*size*size + m*size + n] = buffer[i*channels*size*size + j*size*size + m*size + n] - average;
                    }
                }/*
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
                    }
                }*/

            }
        }
        delete[] buffer;
    }
    fin.close();
}

void Cots::initWeight(int me, bool type)
{
    int length = this->_pars->filter_size*this->_pars->filter_size*this->_pars->filter_channels*this->_pars->input_channels*this->_pars->block_size*this->_pars->block_size;
    if(type)
    {
        for(int process_idx = 0; process_idx < this->_pars->process_num; process_idx++)
        {
            for(int i = 0; i < length; i++)
            {   
                this->_pars->block_weight[i] = RandomWeight();
            }     
            buildWeight(me, this->_pars->block_weight, this->_pars->weight, process_idx, true);
        }
    }
    else
    {
        //读取文件中weight
        string weight_name;
        stringstream ss;
        ss << me;
        ss >> weight_name;
        weight_name = "./binary/weight/layer1_" + weight_name + ".bin";
        ifstream fin(weight_name.c_str(), ios::binary);
        fin.read((char *)this->_pars->weight, length*sizeof(float));       
    }
}

void Cots::filterLayer(int me, int batch_idx)
{
    int r_size = this->_pars->input_size*this->_pars->input_size*this->_pars->input_channels*this->_pars->batch_size;
    int h_size = this->_pars->out_size*this->_pars->out_size*this->_pars->filter_channels*this->_pars->batch_size;
    zeros(this->_pars->send_reconstruct, r_size);
    zeros(this->_pars->send_hidden, h_size);
    zeros(this->_pars->receive_reconstruct, r_size);
    zeros(this->_pars->receive_hidden, h_size);
    for(int process_idx = 0; process_idx < this->_pars->process_num; process_idx++)
    {
        buildR(me, this->_pars->block_input, this->_pars->input, process_idx, false);
        buildWeight(me, this->_pars->block_weight, this->_pars->normalize_weight, process_idx, false); 
        computeH();
        computeR();           
        buildR(me, this->_pars->block_reconstruct, this->_pars->send_reconstruct, process_idx, true);
        buildH(me, this->_pars->block_hidden, this->_pars->send_hidden, process_idx, true); 
    }
    //进行通信，将r全部加在一起，并得到合成后的r
    MPI_Allreduce(this->_pars->send_reconstruct, this->_pars->receive_reconstruct, r_size, MPI_FLOAT, \
            MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(this->_pars->send_hidden, this->_pars->receive_hidden, h_size, MPI_FLOAT, \
            MPI_SUM, MPI_COMM_WORLD);
}

void Cots::poolingLayer(int me, int batch_idx)
{
    int h_size = this->_pars->out_size*this->_pars->out_size*this->_pars->filter_channels*this->_pars->batch_size;
    zeros(this->_pars->send_pooling, h_size); 
    zeros(this->_pars->receive_pooling, h_size); 
    for(int process_idx = 0; process_idx < this->_pars->process_num; process_idx++)
    {
        computeP(me, process_idx, 0);
        buildH(me, this->_pars->block_pooling, this->_pars->send_pooling, process_idx, true);
    }
    MPI_Allreduce(this->_pars->send_pooling, this->_pars->receive_pooling, h_size, MPI_FLOAT, \
            MPI_SUM, MPI_COMM_WORLD);
}

void Cots::lcnLayer(int me, int batch_idx)
{
    int h_size = this->_pars->out_size*this->_pars->out_size*this->_pars->filter_channels*this->_pars->batch_size;
    zeros(this->_pars->send_lcn, h_size);
    zeros(this->_pars->receive_lcn, h_size);
    for(int process_idx = 0; process_idx < this->_pars->process_num; process_idx++)
    {   
        computeLcn(me, process_idx, true);
        buildH(me, this->_pars->block_lcn, this->_pars->send_lcn, process_idx, true);
    } 
    MPI_Allreduce(this->_pars->send_lcn, this->_pars->receive_lcn, h_size, MPI_FLOAT, \
            MPI_SUM, MPI_COMM_WORLD);
    /*       if(me == 5)
             {
             cout << "================reduce===============\n";
             for(int i = 0; i < 200; i++)
             {
    //   cout << this->_pars->send_lcn[i] << " \t" ;
    cout << this->_pars->send_lcn[i]  <<  ":" <<  this->_pars->receive_lcn[i] << " \t" ;
    }
    cout << endl;
    }*/

    zeros(this->_pars->send_lcn, h_size);
    for(int process_idx = 0; process_idx < this->_pars->process_num; process_idx++)
    {   
        computeLcn(me, process_idx, false);
        buildH(me, this->_pars->block_lcn, this->_pars->send_lcn, process_idx, true);
    }   
    MPI_Allreduce(this->_pars->send_lcn, this->_pars->receive_lcn, h_size, MPI_FLOAT, \
            MPI_SUM, MPI_COMM_WORLD);
    /*     if(me == 0)
           {
           for(int i = 0; i < 200; i++)
           {
           cout << this->_pars->receive_hidden[i]  << ":" << this->_pars->receive_pooling[i] <<  ":" << this->_pars->receive_lcn[i] << " \t" ;
           }
           cout << endl;
           }*/
}

void Cots::computeH()
{
    int m = this->_pars->filter_channels*this->_pars->block_size*this->_pars->block_size,
        n = this->_pars->batch_size,
        k = this->_pars->filter_size*this->_pars->filter_size*this->_pars->input_channels,
        lda = this->_pars->filter_size*this->_pars->filter_size*this->_pars->input_channels,
        ldb = n,
        ldc = this->_pars->batch_size;
    float alpha = this->_pars->alpha,
          beta = 0;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, this->_pars->block_weight, lda, this->_pars->block_input, ldb, beta, this->_pars->block_hidden, ldc);
}

void Cots::buildH(int me, float *block, float *all, int process_idx, bool ward)
{
    int id = process_idx*this->_pars->thread_num + me;
    int row_block = id/(this->_pars->out_size/this->_pars->step);
    int col_block = id%(this->_pars->out_size/this->_pars->step);
    for(int i = 0; i < this->_pars->batch_size; i++)
    {
        for(int j = 0; j < this->_pars->filter_channels; j++)
        {
            int start = i*this->_pars->filter_channels*this->_pars->out_size*this->_pars->out_size  \
                        + j*this->_pars->out_size*this->_pars->out_size  \
                        + row_block*this->_pars->step*this->_pars->out_size \
                        + col_block*this->_pars->step;
            for(int m = 0; m < this->_pars->block_size; m++)
            {
                for(int n = 0; n < this->_pars->block_size; n++)
                {
                    int pos = (j*this->_pars->block_size*this->_pars->block_size + m*this->_pars->block_size + n)*this->_pars->batch_size + i;
                    if(ward)
                    {
                        all[start + m*this->_pars->out_size + n] = block[pos];
                    }
                    else
                    {
                        block[pos] = all[start + m*this->_pars->out_size + n];
                    }
                }
            }
        }
    }
}

void Cots::computeR()
{
    int m = this->_pars->filter_size*this->_pars->filter_size*this->_pars->input_channels,
        n = this->_pars->batch_size,
        k = this->_pars->filter_channels*this->_pars->block_size*this->_pars->block_size,
        lda = this->_pars->input_channels*this->_pars->filter_size*this->_pars->filter_size,
        ldb = this->_pars->batch_size,
        ldc = this->_pars->batch_size;
    float alpha = 1,
          beta = 0;
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m, n, k, alpha, this->_pars->block_weight, lda, this->_pars->block_hidden, ldb, beta, this->_pars->block_reconstruct, ldc);
}

void Cots::buildR(int me, float *block, float *all, int process_idx, bool ward)
{
    for(int i = 0; i < this->_pars->batch_size; i++)
    {                    
        for(int j = 0; j < this->_pars->input_channels; j++)
        {
            int id = process_idx*this->_pars->thread_num + me;
            int row_block = id/(this->_pars->out_size/this->_pars->step);
            int col_block = id%(this->_pars->out_size/this->_pars->step);
            int start = i*this->_pars->input_channels*this->_pars->input_size*this->_pars->input_size \
                        + j*this->_pars->input_size*this->_pars->input_size  \
                        + row_block*this->_pars->step*this->_pars->input_size \
                        + col_block*this->_pars->step;
            for(int m = 0; m < this->_pars->filter_size; m++)
            {
                for(int n = 0; n < this->_pars->filter_size; n++)
                {
                    int pos = (j*this->_pars->filter_size*this->_pars->filter_size + m*this->_pars->filter_size + n)*this->_pars->batch_size + i;
                    if(ward)
                    {
                        //将block_r还原到r矩阵
                        all[start + m*this->_pars->input_size + n] += block[pos];
                    }
                    else
                    {
                        block[pos] = all[start + m*this->_pars->input_size + n];
                    }
                }
            }
        }
    }
}

void Cots::buildWeight(int me, float *block, float *all, int process_idx, bool ward)
{
    int length = this->_pars->filter_size*this->_pars->filter_size*this->_pars->filter_channels*this->_pars->input_channels*this->_pars->block_size*this->_pars->block_size;
    int start = process_idx*length;
    for(int m = 0; m < length; m++)
    {
        if(ward)
        {
            all[start + m] = block[m];
        }
        else
        {
            block[m] = all[start + m];
        }
    }
}

void Cots::normalizeWeight()
{
    //将所有的weight都normalize
    int length1 = this->_pars->process_num*this->_pars->filter_channels*this->_pars->block_size*this->_pars->block_size;
    for(int m = 0; m < length1; m++)
    {
        int length2 = this->_pars->filter_size*this->_pars->filter_size*this->_pars->input_channels;
        float sum = 0.0001;
        for(int i = 0; i < length2; i++)
        {
            sum += this->_pars->weight[i + m*length2]*this->_pars->weight[i + m*length2];
        }
        for(int i = 0; i < length2; i++)
        {
            this->_pars->normalize_weight[i + m*length2] = this->_pars->weight[i + m*length2]/sqrt(sum);
        }
    }
}

void Cots::inverseProjWeight(float *graident, float *origin_weight, float *project_weight)
{
    int length1 = this->_pars->filter_channels*this->_pars->block_size*this->_pars->block_size;
    for(int m = 0; m < length1; m++)
    {
        int length2 = this->_pars->filter_size*this->_pars->filter_size*this->_pars->input_channels;
        float sum = 0.0001;
        float sum1 = 0;
        for(int i = 0; i < length2; i++)
        {
            sum += origin_weight[i + m*length2]*origin_weight[i + m*length2];
            sum1 += origin_weight[i + m*length2]*graident[i + m*length2];
        }
        for(int i = 0; i < length2; i++)
        {
            graident[i + m*length2] = graident[i + m*length2]/sqrt(sum) - project_weight[i + m*length2]*sum1/sum;
        }
    }
}

void Cots::computeP(int me, int process_idx, int type)
{
    //type表示是计算pooling，还是h/p
    for(int i = 0; i < this->_pars->batch_size; i++)
    {
        for(int j = 0; j < this->_pars->filter_channels; j++)
        {
            //block的起点
            int id = process_idx*this->_pars->thread_num + me;
            int row_block = id/(this->_pars->out_size/this->_pars->step);
            int col_block = id%(this->_pars->out_size/this->_pars->step);
            int start_in_receive = i*this->_pars->filter_channels*this->_pars->out_size*this->_pars->out_size  \
                                   + j*this->_pars->out_size*this->_pars->out_size  \
                                   + row_block*this->_pars->step*this->_pars->out_size \
                                   + col_block*this->_pars->step;
            for(int m = 0; m < this->_pars->block_size; m++)
            {
                for(int n = 0; n < this->_pars->block_size; n++)
                {
                    float sum = 0;
                    //计算周围九个点的平方和
                    for(int k = -1; k < this->_pars->pooling_size - 1; k++)
                    {
                        for(int t = -1; t < this->_pars->pooling_size - 1; t++)
                        {
                            //当点在hidden内时才加到sum中
                            if((col_block + n + t >= 0)&&(col_block + n + t <= this->_pars->out_size/this->_pars->block_size)       \
                                    &&(row_block + m + k >= 0)&&(row_block + m + k <= this->_pars->out_size/this->_pars->block_size))
                            {
                                //type, 0代表计算pooling， 1代表计算delta_w, 2代表计算delta_alpha
                                switch(type)
                                {
                                    case 0:
                                        {
                                            sum += this->_pars->receive_hidden[start_in_receive + (m+k)*this->_pars->out_size + \
                                                   t+n]*this->_pars->receive_hidden[start_in_receive + (m+k)*this->_pars->out_size + t+n];
                                            break;
                                        }
                                    case 1:
                                        {
                                            if(this->_pars->receive_pooling[start_in_receive + (m+k)*this->_pars->out_size + n+t] != 0)
                                            {
                                                //      sum += this->_pars->receive_hidden[start_in_receive + (k+m)*this->_pars->block_size + \
                                                //           t + n]/this->_pars->receive_pooling[start_in_receive + m*this->_pars->out_size + n];
                                                sum += this->_pars->receive_hidden[start_in_receive + m*this->_pars->block_size + \
                                                       n]/this->_pars->receive_pooling[start_in_receive + (m+k)*this->_pars->out_size + n+t];
                                            }
                                            break;
                                        }
                                    case 2:
                                        {
                                            if(this->_pars->receive_pooling[start_in_receive + (m+k)*this->_pars->out_size + n+t] != 0)
                                            {
                                                //   sum += this->_pars->receive_hidden[start_in_receive + (k+m)*this->_pars->out_size + \
                                                //          t + n]*this->_pars->receive_hidden[start_in_receive + (k+m)*this->_pars->out_size + \
                                                //          t + n]/this->_pars->receive_pooling[start_in_receive + m*this->_pars->out_size + n];
                                                sum += this->_pars->receive_hidden[start_in_receive + m*this->_pars->out_size + \
                                                       n]*this->_pars->receive_hidden[start_in_receive + m*this->_pars->out_size + \
                                                       n]/this->_pars->receive_pooling[start_in_receive + (m+k)*this->_pars->out_size + n+t];
                                            }
                                            break;
                                        }
                                }
                            }
                        }
                    }
                    int pos_in_block = (j*this->_pars->block_size*this->_pars->block_size + m*this->_pars->block_size + n)*this->_pars->batch_size + i;
                    switch(type)
                    {
                        case 0:
                            {
                                this->_pars->block_pooling[pos_in_block] = sqrt(sum); 
                                break;
                            }
                        case 1:
                            {
                                this->_pars->block_pooling[pos_in_block] = sum; 
                                break;
                            }
                        case 2:
                            {
                                this->_pars->block_pooling[pos_in_block] = sum;
                                break;
                            }
                    }
                }
            }
        }
    }
}

void Cots::computeLcn(int me, int process_idx, bool type)
{
    //type表示是计算减法，还是计算除法
    for(int i = 0; i < this->_pars->batch_size; i++)
    {
        float *sum1 = new float[this->_pars->block_size*this->_pars->block_size];
        float *sum2 = new float[this->_pars->block_size*this->_pars->block_size];       
        zeros(sum1, this->_pars->block_size*this->_pars->block_size);
        zeros(sum2, this->_pars->block_size*this->_pars->block_size);
        for(int j = 0; j < this->_pars->filter_channels; j++)
        {
            //block的起点
            int id = process_idx*this->_pars->thread_num + me;
            int row_block = id/(this->_pars->out_size/this->_pars->step);
            int col_block = id%(this->_pars->out_size/this->_pars->step);
            int start = i*this->_pars->filter_channels*this->_pars->out_size*this->_pars->out_size  \
                        + j*this->_pars->out_size*this->_pars->out_size  \
                        + row_block*this->_pars->step*this->_pars->out_size \
                        + col_block*this->_pars->step;
            float gaussion[9] = {0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625};
            for(int m = 0; m < this->_pars->block_size; m++)
            {
                for(int n = 0; n < this->_pars->block_size; n++)
                {                    
                    //计算八张图周围九个点的平方和
                    for(int k = -1; k < this->_pars->pooling_size - 1; k++)
                    {
                        for(int t = -1; t < this->_pars->pooling_size - 1; t++)
                        {
                            //当点在hidden内时才加到sum中
                            if((col_block + n + t >= 0)&&(col_block + n + t <= this->_pars->out_size/this->_pars->block_size)       \
                                    &&(row_block + m + k >= 0)&&(row_block + m + k <= this->_pars->out_size/this->_pars->block_size))
                            {
                                if(type)
                                    sum1[m*this->_pars->block_size + n] += gaussion[(k+1)*this->_pars->pooling_size + \
                                                                           t + 1]*this->_pars->receive_pooling[start + (k+m)*this->_pars->out_size + t+n]/this->_pars->filter_channels;
                                else
                                {
                                    sum2[m*this->_pars->block_size + n] += gaussion[(k+1)*this->_pars->pooling_size + t \
                                                                           + 1]*this->_pars->receive_lcn[start + (k+m)*this->_pars->out_size + t+n] \
                                                                           *this->_pars->receive_lcn[start + (m+k)*this->_pars->out_size + t+n]/this->_pars->filter_channels;
                                }
                            }
                        }
                    }
                }
            }
        }
        for(int j = 0; j < this->_pars->filter_channels; j++)
        {
            int id = process_idx*this->_pars->thread_num + me;
            int row_block = id/(this->_pars->out_size/this->_pars->step);
            int col_block = id%(this->_pars->out_size/this->_pars->step);
            int start = i*this->_pars->filter_channels*this->_pars->out_size*this->_pars->out_size  \
                        + j*this->_pars->out_size*this->_pars->out_size  \
                        + row_block*this->_pars->step*this->_pars->out_size \
                        + col_block*this->_pars->step; 
            for(int m = 0; m < this->_pars->block_size; m++)
            {
                for(int n = 0; n < this->_pars->block_size; n++)
                {       	
                    //subtractive normalizations
                    int pos_in_block = (j*this->_pars->block_size*this->_pars->block_size + m*this->_pars->block_size + n)*this->_pars->batch_size + i;
                    if(type)
                    {
                        this->_pars->block_lcn[pos_in_block] = this->_pars->receive_pooling[start + m*this->_pars->out_size + n] \
                                                               - sum1[m*this->_pars->block_size + n];
                        //      if(me == 5)
                        //        cout << this->_pars->block_lcn[pos_in_block] << ":" << sum1[m*this->_pars->block_size + n] << "\t";

                    }
                    else
                    {
                        this->_pars->block_lcn[pos_in_block] = this->_pars->receive_lcn[start + m*this->_pars->out_size + n] \
                                                               /(sqrt(sum2[m*this->_pars->block_size + n]) > 0.01 ? sqrt(sum2[m*this->_pars->block_size + n]) : 0.01);


                        //        if(me == 0)
                        //          cout << this->_pars->block_lcn[pos_in_block] << ":" << this->_pars->receive_lcn[start + m*this->_pars->out_size + n] << "\t";
                    }//更新下一个点
                }
            }
        }
        delete[] sum1;
        delete[] sum2;
    }
}

void Cots::updateW(int me, int batch_idx)
{
    int length = this->_pars->filter_size*this->_pars->filter_size*this->_pars->filter_channels*this->_pars->input_channels*this->_pars->block_size*this->_pars->block_size;
    float *block_dw1 = new float[length]; 
    float *block_dw2 = new float[length];
    float *block_winc = new float[length];
    float *block_origin_weight = new float[length];
    zeros(block_dw1, length);
    zeros(block_dw2, length);
    zeros(block_winc, length);
    zeros(block_origin_weight, length);
    for(int process_idx = 0; process_idx < this->_pars->process_num; process_idx++)
    {       	
        buildR(me, this->_pars->block_input, this->_pars->input, process_idx, false);
        buildWeight(me, this->_pars->block_weight, this->_pars->normalize_weight, process_idx, false);
        buildWeight(me, block_origin_weight, this->_pars->weight, process_idx, false);
        buildWeight(me, block_winc, this->_pars->winc, process_idx, false);
        buildH(me, this->_pars->block_hidden, this->_pars->receive_hidden, process_idx, false);
        buildR(me, this->_pars->block_reconstruct, this->_pars->receive_reconstruct, process_idx, false);

   /*     if(me == 0&&process_idx == 0)
        {
            cout << "=========delta_w1, delta_w2=========\n";
            for(int i = 0; i < 10; i++)
            {
                cout << this->_pars->block_hidden[i]  << ":" << this->_pars->block_weight[i] << ":" << block_origin_weight[i]<< ":"<< this->_pars->block_reconstruct[i]<< "\t";
            }
            cout << endl;
        }*/
        int h_length = this->_pars->filter_channels*this->_pars->block_size*this->_pars->block_size*this->_pars->batch_size;
        zeros(this->_pars->block_pooling, h_length);
        //计算第一层的dw
        computeDw1(block_dw1);
        //计算第二层的dw
        computeP(me, process_idx, 1);
        computeDw2(block_dw2);
        //更新权重
        if(me == 0&&process_idx == 0)
        {
            cout << "=========delta_w1, delta_w2=========\n";
            for(int i = 0; i < 10; i++)
            {
                cout << block_dw1[i]  << ":" << block_dw2[i] << "\t";
            }
            cout << endl;
        }
        catlas_saxpby(length, 2, block_dw1, 1, 1, block_dw2, 1);
        //将dw反向投影回原平面
        inverseProjWeight(block_dw2, block_origin_weight, this->_pars->block_weight);
        catlas_saxpby(length, this->_pars->momentum, block_winc, 1, this->_pars->learning_rate, block_dw2, 1);
        catlas_saxpby(length, 1, block_dw2, 1, 1, block_origin_weight, 1);
        buildWeight(me, block_origin_weight, this->_pars->weight, process_idx, true);
        buildWeight(me, block_dw2, this->_pars->winc, process_idx, true);
        zeros(this->_pars->block_pooling, h_length);
        computeP(me, process_idx, 2);
        float sum = 0;
        for(int i = 0; i < h_length; i++)
        {
            sum += this->_pars->block_pooling[i];
        }
        this->_delta_alpha += this->_lambda*sum/this->_pars->alpha;
    }
    float send_delta_alpha = this->_delta_alpha/this->_pars->batch_size;
    MPI_Allreduce(&send_delta_alpha, &this->_delta_alpha, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    if(me == 0)
    {
        cout << "=========weight=========\n";
        for(int i = 0; i < 100; i++)
        {
            cout << block_origin_weight[i] << "\t";
        }
        cout << endl;
        cout << "=========delta_weight=========\n";
        for(int i = 0; i < 10; i++)
        {
            cout << block_dw2[i] << "\t";
        }
        cout << endl;
    }

    delete[] block_dw1;
    delete[] block_dw2;
    delete[] block_winc;
    delete[] block_origin_weight;
}

void Cots::updateAlpha()
{
    int r_all_length = this->_pars->input_size*this->_pars->input_size*this->_pars->input_channels*this->_pars->batch_size;
    float sum = 0;
    for(int i = 0; i < r_all_length; i++)
    {
        sum += 2*(this->_pars->receive_reconstruct[i] - this->_pars->input[i])*this->_pars->receive_reconstruct[i];
    }
    this->_delta_alpha += sum/(this->_pars->alpha*this->_pars->batch_size);
}



void Cots::computeDw1(float *block_dw1)
{

    int block_r_size = this->_pars->batch_size*this->_pars->filter_size*this->_pars->filter_size*this->_pars->input_channels;
    //计算r-x，存在r里面
    catlas_saxpby(block_r_size, -1, this->_pars->block_input, 1, 1, this->_pars->block_reconstruct, 1);
    int block_w_size = this->_pars->filter_size*this->_pars->filter_size*this->_pars->input_channels*this->_pars->block_size*this->_pars->block_size*this->_pars->filter_channels;
    int block_w_r_size = this->_pars->block_size*this->_pars->block_size*this->_pars->filter_channels*this->_pars->batch_size;
    float *block_h_r = new float[block_w_size];
    float *block_w_r = new float[block_w_r_size];
    zeros(block_h_r, block_w_size);
    zeros(block_w_r, block_w_r_size);
    //计算h*(r-x)'
    int m = this->_pars->block_size*this->_pars->block_size*this->_pars->filter_channels,
        n = this->_pars->input_channels*this->_pars->filter_size*this->_pars->filter_size,
        k = this->_pars->batch_size,
        lda = this->_pars->batch_size,
        ldb = this->_pars->batch_size,
        ldc = this->_pars->input_channels*this->_pars->filter_size*this->_pars->filter_size;
    float alpha = 1,
          beta = 0;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, alpha, this->_pars->block_hidden, lda, this->_pars->block_reconstruct, ldb, beta, block_h_r, ldc);
    //计算w*(r-x)
    m = this->_pars->block_size*this->_pars->block_size*this->_pars->filter_channels,
      n = this->_pars->batch_size,
      k = this->_pars->input_channels*this->_pars->filter_size*this->_pars->filter_size,
      lda = this->_pars->input_channels*this->_pars->filter_size*this->_pars->filter_size,
      ldb = this->_pars->batch_size,
      ldc = this->_pars->batch_size;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, this->_pars->block_weight, lda, this->_pars->block_reconstruct, ldb, beta, block_w_r, ldc);
    //计算w*(r-x)*x'
    m = this->_pars->block_size*this->_pars->block_size*this->_pars->filter_channels,
      n = this->_pars->input_channels*this->_pars->filter_size*this->_pars->filter_size,
      k = this->_pars->batch_size,
      lda = this->_pars->batch_size,
      ldb = this->_pars->batch_size,
      ldc = this->_pars->input_channels*this->_pars->filter_size*this->_pars->filter_size;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, this->_pars->alpha, block_w_r, lda, this->_pars->block_input, ldb, beta, block_dw1, ldc);
    //计算h*(r-x)'+ alpha*w*(r-x)*x'
    catlas_saxpby(block_w_size, 1, block_h_r, 1, 1, block_dw1, 1);

    delete[] block_h_r;
    delete[] block_w_r;
}

void Cots::computeDw2(float *block_dw2)
{
    //计算alpha*h/p*x'
    int m = this->_pars->block_size*this->_pars->block_size*this->_pars->filter_channels,
        n = this->_pars->input_channels*this->_pars->filter_size*this->_pars->filter_size,
        k = this->_pars->batch_size,
        lda = this->_pars->batch_size,
        ldb = this->_pars->batch_size,
        ldc = this->_pars->input_channels*this->_pars->filter_size*this->_pars->filter_size;
    float alpha = this->_lambda*this->_pars->alpha,
          beta = 0;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, alpha, this->_pars->block_pooling, lda, this->_pars->block_input, ldb, beta, block_dw2, ldc);
}


void Cots::saveWeight(string filename)
{
    int length = this->_pars->process_num*this->_pars->filter_size*this->_pars->filter_size*this->_pars->input_channels*this->_pars->filter_channels*this->_pars->block_size*this->_pars->block_size;
    subSaveFile(filename.c_str(), length, this->_pars->weight, true);
}

/*void Cots::saveOutput()
  {
  int length = this->_pars->input_size*this->_pars->input_size*this->_pars->input_channels*this->_pars->batch_size;
  subSaveFile("./binaryfile/reconstruct.bin", length, this->_pars->receive_reconstruct, true);
//    length = this->_pars->out_size*this->_pars->out_size*this->_pars->filter_channels*this->_pars->batch_size;
//    subSaveFile(this->_output_address, length, this->_pars->receive_lcn, false);
}*/

void Cots::subSaveFile(string filename, int length, float *data, bool type)
{
    ofstream fout;
    if(type)
        fout.open(filename.c_str(), ios::binary);
    else
        fout.open(filename.c_str(), ios::binary|ios::app|ios::out);
    fout.write((char*)data, length*sizeof(float));
    fout.close();

}

void Cots::zeros(float *all, int length)
{
    for(int i = 0; i < length; i++)
    {
        all[i] = 0;
    }
}

void Cots::assignMemory()
{
    //1.初始化w
    //weight大小为2*2*8*10*10*3，列为2*2*8，行为10*10*3
    int length = this->_pars->filter_size*this->_pars->filter_size*this->_pars->filter_channels*this->_pars->input_channels*this->_pars->block_size*this->_pars->block_size;
    this->_pars->block_weight = new float[length];
    int weight_length = this->_pars->process_num*this->_pars->filter_size*this->_pars->filter_size*this->_pars->input_channels*this->_pars->filter_channels*this->_pars->block_size*this->_pars->block_size;
    this->_pars->weight = new float[weight_length];
    this->_pars->normalize_weight = new float[weight_length];
    zeros(this->_pars->block_weight, length);
    zeros(this->_pars->weight, weight_length);
    zeros(this->_pars->normalize_weight, weight_length);
    //2.初始化x
    int block_input_size = this->_pars->batch_size*this->_pars->filter_size*this->_pars->filter_size*this->_pars->input_channels;
    this->_pars->block_input = new float[block_input_size];     
    this->_pars->block_reconstruct = new float[block_input_size];
    zeros(this->_pars->block_input, block_input_size);
    zeros(this->_pars->block_reconstruct, block_input_size);

    int block_hidden_size = this->_pars->filter_channels*this->_pars->block_size*this->_pars->block_size*this->_pars->batch_size;
    this->_pars->block_hidden = new float[block_hidden_size];
    this->_pars->block_pooling = new float[block_hidden_size]; 
    this->_pars->block_lcn = new float[block_hidden_size];
    zeros(this->_pars->block_hidden, block_hidden_size);
    zeros(this->_pars->block_pooling, block_hidden_size);
    zeros(this->_pars->block_lcn, block_hidden_size);
    //3.初始化最后的r
    int r_size = this->_pars->input_size*this->_pars->input_size*this->_pars->input_channels*this->_pars->batch_size;
    this->_pars->input = new float[r_size];
    this->_pars->send_reconstruct = new float[r_size];
    this->_pars->receive_reconstruct = new float[r_size];
    zeros(this->_pars->input, r_size);
    zeros(this->_pars->send_reconstruct, r_size);
    zeros(this->_pars->receive_reconstruct, r_size);
    //4.初始化最后的h
    int h_size = this->_pars->out_size*this->_pars->out_size*this->_pars->filter_channels*this->_pars->batch_size;
    this->_pars->send_hidden = new float[h_size];
    this->_pars->receive_hidden = new float[h_size];
    this->_pars->send_pooling = new float[h_size];
    this->_pars->receive_pooling = new float[h_size];
    this->_pars->send_lcn = new float[h_size];
    this->_pars->receive_lcn = new float[h_size];
    zeros(this->_pars->send_hidden, h_size);
    zeros(this->_pars->receive_hidden, h_size);
    zeros(this->_pars->send_pooling, h_size);
    zeros(this->_pars->receive_pooling, h_size);
    zeros(this->_pars->send_lcn, h_size);
    zeros(this->_pars->receive_lcn, h_size);
}

void Cots::clearMemory()
{
    delete[] this->_pars->block_input;
    delete[] this->_pars->block_weight;
    delete[] this->_pars->block_hidden;
    delete[] this->_pars->block_reconstruct;
    delete[] this->_pars->block_pooling;
    delete[] this->_pars->block_lcn;
    delete[] this->_pars->input;
    delete[] this->_pars->send_reconstruct;
    delete[] this->_pars->receive_reconstruct;
    delete[] this->_pars->send_hidden;
    delete[] this->_pars->receive_hidden;
    delete[] this->_pars->send_pooling;
    delete[] this->_pars->receive_pooling;
    delete[] this->_pars->send_lcn;
    delete[] this->_pars->receive_lcn;
    delete[] this->_pars->weight;
    delete[] this->_pars->normalize_weight;
    delete this->_pars;
}





























