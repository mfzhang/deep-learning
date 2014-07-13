/*************************************************************************
    > File Name: matrix.cpp
    > Author: chenrudan
    > Mail: chenrudan123@gmail.com
    > Created Time: 2014年07月13日 星期日 16时08分16秒
 ************************************************************************/

#include<iostream>
#include<vector>
#include"matrix.h"

using namespace std;

Matrix::Matrix(int row, int col)
{
	this->row_ = row;
	this->col_ = col;
	row_unfull_pos_ = 0;
	col_unfull_pos_ = 0;

	//为二维数组分配内存
	all_element_ = new double*[col];
	for(int i = 0; i < this->col_; i++)
	{
		all_element_[i] = new double[row];
	}
}

Matrix::~Matrix()
{
	for(int i = 0; i < this->col_; i++)
	{
		delete[] all_element_[i];
	}
	delete[] all_element_;

}

Matrix Matrix::MatrixMultiply(Matrix &mat_1, Matrix &mat_2)
{
	long prod_row = mat_1.GetRowNum();
	long prod_col = mat_2.GetColNum();
	Matrix prod_mat(prod_row, prod_col);

	for(long i = 0; i < prod_row; i++)
	{
		for(long j = 0; j < prod_col; j++)
		{
			for(long k = 0; k < mat_1.GetColNum(); k++)
			{
				prod_mat.all_element_[i][j] += mat_1.all_element_[i][k]*mat_2.all_element_[k][j];
			}
		}
	}
	return prod_mat;
}

double Matrix::GetElement(long row, long col)
{
	return all_element_[row][col];
}

long Matrix::GetRowNum()
{
	return this->row_;
}

long Matrix::GetColNum()
{
	return this->col_;
}



void Matrix::AddElement(double element)
{
	all_element_[row_unfull_pos_][col_unfull_pos_] = element;
	col_unfull_pos_++;
	if(col_unfull_pos_ >= this->col_)
	{
		row_unfull_pos_++;
		col_unfull_pos_ = 0;
	}
	if(row_unfull_pos_ >= this->row_)
	{
		cout << "you have enter two many element!\n" ;
		row_unfull_pos_--;
	}
}





































































