#include <iostream>
#include <mpi.h>
#include "load.h"

using namespace std;

void managerNode(int me, float *image);
void workerNode(int me, float *image);

struct Pars{
    int input_channels;
    int filter_size;
    int filter_channels;
};

int main(int argc, char **argv)
{
    int nnode, //进程数，从命令行读取
        me;    //MPI ID
    float *image;
    float *weight;  //在每一个进程里面weight取值或者范围都不一样
    int batch_size = 48;
    Pars *par;
    par->filter_channels = 8;
    par->filter_size = 10;
    par->input_channels = 3;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nnode);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    Load load;
    for(int batch_idx = 0; batch_idx < batch_size; batch_idx++)
    {
        image = load.LoadPartData("../../../crd/deeplearning/data/stl10/train_X.bin",  \
                                  batch_size*(me - 1)*96*96*3*(5000/nnode) + batch_idx*nnode*96*96*3*batch_size,
                                  batch_size*me*96*96*3*(5000/nnode) + batch_idx*nnode*96*96*3*batch_size);
        if(me == 0)
            managerNode(me, image, weight, pars);
        else
            workerNode(me, image, weight, pars);
    }
    MPI_Finalize();
    return 0;
}

void managerNode(int me, float *image, float *weight, struct Pars pars)
{

}

void workerNode(int me, float *image, float *weight, struct Pars pars)
{
    //1.计算wx
    initWeight(weight, pars);
}

void initWeight(float *weight, struct Pars *pars)
{
    float high = 4 * sqrt(6.0 / (2 * pars->filter_size * pars->filter_size * pars->filter_channels));
    float low = -high;
}








