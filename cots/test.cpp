/* Filename: main.cpp
 * -------------------
 * 这个文件打开一个二进制文件并传入不同线程跟权重相乘
 */

#include <iostream>
#include <cmath>
#include "mpi.h"
#include "utils.h"
#include "load.h"
extern "C"{
#include "cblas.h"
}

using namespace std;

void managerNode(int me, struct Pars *pars);
void workerNode(int me, struct Pars *pars);
void getBlockInput(struct Pars *pars);

struct Pars{
    int input_channels;
    int input_size;
    int filter_size;
    int filter_channels;
    int batch_size;
    int block_size;
    int step;
    //zero pass to other thread
    float *block_input;
    //weight for block compute
    float *block_weight;
    float *block_hidden;
    float *block_reconstruct;
    //r for zero
    float *reconstruct;
    //weight for zero
    float *weight;
    //input for zero
    float *input;
};

int main(int argc, char **argv)
{
    int nnode, //进程数，从命令行读取
        me;    //MPI ID
    struct Pars *pars = new Pars;
    pars->filter_channels = 8;
    pars->filter_size = 8;
    pars->input_channels = 3;
    pars->input_size = 96;
    pars->batch_size = 48;
    pars->block_size = 2;
    pars->step = 2;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nnode);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);

    cout << me << endl;

    if(me == 0)
        managerNode(me, pars);
    else
        workerNode(me, pars);

    MPI_Finalize();

    return 0;
}

/* Function: managerNode
 * --------------------
 * Download the input data, and separate these data and send to other
 * thread to compute the hidden and the reconstuct result, then receive 
 * the part r to the whole r.
 */
void managerNode(struct Pars *pars)
{
    Load load;
    pars->input = load.loadData("preprocessed.bin");
    int block_input_size = pars->batch_size*pars->filter_size*pars->filter_size*pars->input_channels;
    pars->block_input = new float[block_input_size];
    for(int batch_idx = 0; batch_idx < 48/pars->batch_size; batch_idx++)
    {   
        // 共44个线程，需要运行44次才能得到88*88个点 
        for(int me_idx = 0; me_idx < pars->nnode; me_idx++)
        {
            for(int process_idx = 0; process_idx < 44; process_idx++)
            {
                getBlockInput(me_idx, pars);
                
            }
        }
    }
}

void getBlockInput(int me_idx, struct Pars *pars)
{
    //获取每一次需要用的48*3*10*10
    for(int i = 0; i < pars->batch_size; i++)
    {   
        for(int j = 0; j < pars->input_channels; j++)
        {
            int start = batch_idx*pars->batch_size*pars->input_channels*pars->input_size*pars->input_size \
                        + i*pars->input_channels*pars->input_size*pars->input_size \
                        + j*pars->input_size*pars->input_size  \
                        + (process_idx - 1)*pars->step*pars->input_size  \
                        + me_idx*pars->step*pars->input_size;
            for(int m = 0; m < pars->filter_size; m++)
            { 
                for(int n = 0; n < pars->filter_size; n++)
                {  
                    pars->block_input[i*pars->input_channels*pars->filter_size*pars->filter_size \
                        + j*pars->filter_size*pars->filter_size + m*pars->filter_size + n] = \ 
                        pars->input[start + m*pars->filter_size + n]; 
                }   
            } 
        }   
    } 
}

/* Function: workerNode
 * --------------------
 * receive the block input and compute the hidden then compute the 
 * reconstruct.
 */
void workerNode(int me, struct Pars *pars)
{

}



















































































