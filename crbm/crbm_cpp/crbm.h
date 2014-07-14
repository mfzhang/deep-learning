/*************************************************************************
    > File Name: crbm.h
    > Author: chenrudan
    > Mail: chenrudan123@gmail.com
    > Created Time: 2014年07月14日 星期一 19时57分22秒
 ************************************************************************/

#include <iostream>
#include <vector>
#include "matrix.h"

using namespace std;

class Crbm
{
    public:
        //每一层的结构
        Crbm();
        ~Crbm();



    private:
        struct pars{
            double l2reg;
            //微小项
            double epsilon;
            int num_bases;
            int CD_K;
            int num_channels;
            //存放这一层的所有权重值
            vector<Matrix> weight;
            //对应channel
            vector<double> vbias;
            //对应bases
            vector<double> hbias;
        };

};

