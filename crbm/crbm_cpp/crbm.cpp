/*************************************************************************
    > File Name: crbm.cpp
    > Author: chenrudan
    > Mail: chenrudan123@gmail.com
    > Created Time: 2014年07月14日 星期一 19时57分29秒
 ************************************************************************/

#include <iostream>
#include <vector>
#include <cmath>
#include "crbm.h"
#include "matrix.h"
#include "utils.h"
#include "conv.h"

using namespace std;

Crbm::Crbm()
{
    this->CD_K_ = 1;
    this->weight_ = new vector<Matrix*>();
    this->feature_map_ = new vector<Matrix*>();
    this->first_conv_forward_ = true;
}

Crbm::~Crbm()
{
    delete[] this->weight_;
    delete[] this->feature_map_;
}

void Crbm::FilterInit(int filter_row, int num_channels, int input_channels, int input_row)
{
    this->num_channels_ = num_channels;
    this->filter_row_ = filter_row;
    this->input_channels_ = input_channels;
    this->input_row_ = input_row;
    this->out_size_ = this->input_row_ - this->filter_row_ + 1;

    float high = 4 * sqrt(6.0 / (2 * filter_row * filter_row * num_channels));
    float low = -high;
    for(int i = 0; i < num_channels*input_channels; i++)
    {
        Matrix *tmp_weight = new Matrix(filter_row_, filter_row_);
        this->dw_.push_back(tmp_weight);
        this->pre_dw_.push_back(tmp_weight);
        //初始化权重
        for(int j = 0; j < filter_row * filter_row - 1; j++)
        {
            float rand_value = RandomWeight(low, high);
            tmp_weight->AddElement(rand_value);
        }
        this->weight_->push_back(tmp_weight);

    }
    //初始化vn_sample
    for(int i = 0; i < this->input_channels_; i++)
    {
        Matrix *tmp_vn_sample = new Matrix(this->input_row_, this->input_row_);
        vn_sample_.push_back(tmp_vn_sample);
        //初始化v偏置
        this->vbias_.push_back(0.0);
        this->dvbias_.push_back(0.0);
    }
    //初始化hn_sample
    for(int i = 0; i < this->num_channels_; i++)
    {
        Matrix *tmp_hn_sample = new Matrix(this->out_size_, this->out_size_);
        hn_sample_.push_back(tmp_hn_sample);
        //初始化feature_map
        feature_map_->push_back(tmp_hn_sample);
        //初始化h偏置
        this->hbias_.push_back(0.0);
        this->dhbias_.push_back(0.0);
    }
}

vector<Matrix*>* Crbm::ConvolutionForward(vector<Matrix*> *input_image)
{
    if(this->first_conv_forward_)
    {
        for(int i = 0, k = 0; i < this->num_channels_; i++)
        {
            for(int j = 0; j < this->input_channels_; j++, k++)
            {
                feature_map_->at(i)->MatrixAddNew(Conv::Conv2d(input_image->at(j), weight_->at(k), 1), 1);
            }
            Matrix::MatrixAddBias(feature_map_->at(i), hbias_[i]);
            //采样得到h1
            Sample(feature_map_->at(i));
        }
        this->first_conv_forward_ = false;
        return feature_map_;
    }
    else
    {
        InitPars(hn_sample_);
        //当进行gibbs采样时
        for(int i = 0, k = 0; i < this->num_channels_; i++)
        {
            for(int j = 0; j < this->input_channels_; j++, k++)
            {
                hn_sample_.at(i)->MatrixAddNew( Conv::Conv2d(input_image->at(j), weight_->at(k), 1) , 1);
            }
            Matrix::MatrixAddBias(hn_sample_.at(i), hbias_[i]);
            //采样得到h1
            Sample(hn_sample_.at(i));
        }
        return 0;
    }
}

void Crbm::ConvolutionBackward(vector<Matrix*> *hidden_map)
{
    InitPars(vn_sample_);
    for(int i = 0; i < this->num_channels_; i++)
    {
        int size = this->out_size_ + 2*(filter_row_*2 - 1);
        //将输入的feature map补上两倍的filter-1大小，用这个图来与分别与三个权重做卷积
        Matrix *recover_hn_sample = new Matrix(size, size);
        for(int m = 9; m < size - 9; m++)
        {
            for(int n = 9; n < size - 9; n++)
            {
                recover_hn_sample->AddElement(hidden_map->at(i)->GetElement(m - 9, n - 9));
            }
        }
        //与三个权重做卷积
        for(int j = 0 ; j < this->input_channels_; j++)
        {
            vn_sample_.at(j)->MatrixAddNew(Conv::Conv2d(recover_hn_sample, weight_->at(i*(this->input_channels_) + j), 1) , 1);
        }
    }
    for(int i = 0; i < input_channels_; i++)
    {
        Matrix::MatrixAddBias(vn_sample_.at(i), vbias_[i]);
        Sample(vn_sample_.at(i));
    }
}

void Crbm::InitPars(vector<Matrix*> &paras)
{
    int length = paras.size();
    int para_size = paras.at(0)->GetRowNum();
    paras.clear();
    for(int i = 0; i < length; i++)
    {
        Matrix *new_para = new Matrix(para_size, para_size);
        paras.push_back(new_para);
    }
}


void Crbm::Sample(Matrix *mat)
{
    float mean;
    for(int i = 0; i < mat->GetRowNum(); i++)
    {
        for(int j = 0; j < mat->GetColNum(); j++)
        {
            mean = Logisitc(mat->GetElement(i, j));
            if(mean < 0 || mean > 1)
                mean = 0.0;
            else
            {
                if(CompareFloat(mean, RandomNumber()))
                    mat->ChangeElement(i, j, 1.0);
                else
                    mat->ChangeElement(i, j, 0.0);
            }
        }
    }
}

void Crbm::ComputeDerivative(vector<Matrix*> *input_image)
{
    //一张输入图和一张输出图来更新一组w，w共24*3组
    InitPars(this->dw_);
    InitPars(this->pre_dw_);
    for(int i = 0; i < this->num_channels_; i++)
    {
        for(int j = 0; j < this->input_channels_; j++)
        {
            dw_.at(i*input_channels_ + j) = Matrix::MatrixAdd( Conv::Conv2d(input_image->at(j), feature_map_->at(i) , 1), 1,  \
                              Conv::Conv2d(vn_sample_.at(j), hn_sample_.at(i), 1), -1);
            dw_.at(i*input_channels_ + j)->MatrixMulCoef(1.0/(this->out_size_*this->out_size_));
            dw_.at(i*input_channels_ + j)->MatrixAddNew(weight_->at(i*input_channels_ + j), -l2reg_);
     //       dw_.at(i*input_channels_ + j)->MatrixMulCoef(this->momentum_);
      //      dw_.at(i*input_channels_ + j)->MatrixAddNew(pre_dw_.at(i*input_channels_ + j), this->epsilon_);
            //将dw的内容拷贝到pre_dw
            for(int m = 0; m < this->filter_row_; m++)
            {
                for(int n = 0; n < this->filter_row_; n++)
                {
                    pre_dw_.at(i*input_channels_ + j)->ChangeElement(m, n, dw_.at(i*input_channels_ + j)->GetElement(m, n));
                }
            }
        }
        //计算dh


    }
}














