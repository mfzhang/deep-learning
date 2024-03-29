/* Filename: layer1.cpp
 * -------------------
 * 
 */

#include <iostream>
#include <fstream>
#include <cmath>
#include <time.h>
#include <string>
#include <sstream>
#include "mpi.h"
#include "utils.h"
#include "load.h"
#include "cots.h"
extern "C"{
#include "cblas.h"
}

using namespace std;

void managerNode(int me, Cots layer1);
void workerNode(int me, Cots layer1);

int epoch = 10;

int all_size = 4000;
string weight_name;

int main(int argc, char **argv)
{
    int nnode, //进程数，从命令行读取
        me;    //MPI ID
    int layer1_input_size = 96;
    int layer1_input_channels = 3;
    int layer1_filter_channels = 8;
    int layer1_filter_size = 10;
    int layer1_batch_size = 40;
    int layer1_block_size = 2;
    int layer1_step = 2;
    int layer1_thread_num = 22;
    int layer1_process_num = 44*44/layer1_thread_num;
    int layer1_pooling_size = 3;
    float layer1_learning_rate = -100;
    float layer1_learning_rate_alpha = -0.0001;
    float layer1_alpha = 0.2;
    float layer1_momentum = -0.005;
    //lambda控制dw1和dw2的区别
    float layer1_lambda = 0.004;


    Cots layer1;
    layer1.init(layer1_input_size, layer1_input_channels, layer1_filter_size, layer1_filter_channels, layer1_batch_size, \
             layer1_block_size, layer1_step, layer1_process_num, layer1_thread_num, layer1_pooling_size, layer1_learning_rate, \
             layer1_learning_rate_alpha, layer1_alpha, layer1_momentum, layer1_lambda, "../../data/stl10/unlabeled_X.bin", "./binaryfile/layer1out_unlabeled.bin");

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nnode);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);

    stringstream ss;
    ss << me;
    ss >> weight_name;
    weight_name = "./binaryfile/weight/layer1_" + weight_name + ".bin";

    if(me == 0)    
    {
        cout << "the learning rate of w is :" << layer1_learning_rate << "\n" << "the lambda is :" << layer1_lambda << endl; 
        cout << "the learning rate of alpha is :" << layer1_learning_rate_alpha << "\n" << "the layer1_alpha is :" << layer1_alpha << endl; 
        cout << "the all size is : " << all_size << "\n"<< "the epoch number is : " << epoch <<endl;
        managerNode(me, layer1);
    }
    else
        workerNode(me, layer1);
    MPI_Finalize();
    return 0;
}

void managerNode(int me, Cots layer1)
{
    clock_t t;
    t = clock();
    layer1.trainModel(me, epoch, all_size, true);
    layer1.saveWeight(weight_name);
    t = clock() - t;
    cout << "this train uses " << (float)t/CLOCKS_PER_SEC << "seconds" << endl;
    layer1.clearMemory();
}

void workerNode(int me, Cots layer1)
{
    layer1.trainModel(me, epoch, all_size, true);
    layer1.saveWeight(weight_name);
    layer1.clearMemory();
}























