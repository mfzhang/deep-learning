#include <iostream>
#include <mpi.h>

#include "load.h"

using namespace std;

void managerNode(int me, float *image);
void workerNode(int me, float *image);

struct Pars{
    int input_channels;
    int input_size;
    int filter_size;
    int filter_channels;
    int batch_size;
    int block_size;
    int step;
};

int main(int argc, char **argv)
{
    int nnode, //进程数，从命令行读取
        me;    //MPI ID
    float *block_input;
    float *weight, *block_weight;  //在每一个进程里面weight取值或者范围都不一样

    Pars *pars;
    pars->filter_channels = 8;
    pars->filter_size = 10;
    pars->input_channels = 3;
    pars->input_size = 200;
    pars->batch_size = 48;
    pars->block_size = 2;
    pars->step = 2;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nnode);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);

    if(me == 0)
        managerNode(me, weight, pars);
    else
        workerNode(me, block_input, block_weight, pars);

    MPI_Finalize();
    return 0;
}

void initInput(int me, float *block_input, struct Pars pars, int batch_idx, int process_idx)
{
    //传入每次需要计算的48*10*10*3
    int block_input_size = pars->batch_size*pars->filter_size*pars->filter_size*pars->input_channels;
    block_input = new float[block_input_size];     //由于输入是间隔产生的，所以需要间隔赋值
    for(int i = 0; i < pars->batch_size; i++)
    {
        for(int j = 0; j < pars->input_channels; j++)
        {
            int start = batch_idx*pars->batch_size*pars->input_channels*pars->input_size*pars->input_size \
                        + i*pars->input_channels*pars->input_size*pars->input_size \
                        + j*pars->input_size*pars->input_size  \
                        + (process_idx - 1)*pars->step*pars->input_size  \
                        + (me - 1)*pars->step*pars->input_size;
            Load load;
            float *tmp_input = load.LoadPartData("../../../crd/deeplearning/data/stl10/train_X.bin", start, \
                                      pars->filter_size, pars->input_size, pars->filter_size);
            for(int m = 0; m < pars->filter_size; m++)
            {
                //m是列，n是行
                for(int n = 0; n < pars->filter_size; n++)
                {
                    //第i列也就是第i个图片获取的数据
                    block_input[i*pars->input_channels*pars->filter_size*pars->filter_size \
                        + j*pars->filter_size*pars->filter_size + m*pars->filter_size + n] =
                        tmp_input[m*pars->filter_size + n];
                }
            }
        }
    }

}

void managerNode(int me, float *weight, struct Pars pars)
{

}

void workerNode(int me, float *block_input, float *block_weight, struct Pars pars)
{
    //1.初始化w
    initWeight(block_weight, pars);
    //2.初始化x
    for(int batch_idx = 0; batch_idx < 5000/pars->batch_size; batch_idx++)
    {
        for(int process_idx = 0; process_idx < 44; process_idx++)
        {
            initInput(me, block_input, pars, batch_idx, process_idx);
        }
    }

}

void initWeight(float *weight, struct Pars *pars)
{
    //weight大小为2*2*8*10*10*3，列为2*2*8，行为10*10*3
    int length = pars->filter_size*pars->filter_size*pars->filter_channels*pars->inputnels*pars->block_size*pars->block_size;
    weight = new float[length];
    float high = 4*sqrt(6.0/(2*pars->filter_size*pars->filter_size*pars->filter_channels));
    float low = -high;
    for(int i = 0; i < length; i++)
    {
        weight[i] = RandomWeight(low, high);
    }
}


























