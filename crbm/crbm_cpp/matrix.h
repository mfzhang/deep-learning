/*************************************************************************
    > File Name: matrix.h
    > Author: chenrudan
    > Mail: chenrudan123@gmail.com 
    > Created Time: 2014年07月12日 星期六 20时25分15秒
 ************************************************************************/

#include<iostream>
#include<vector>

using namespace std;

class Matrix 
{
	public:
		Matrix(int row, int col);
		~Matrix();

/* Function: MatrixMultiply
 * ------------------------
 * 做矩阵快速乘法，输入两个矩阵，得到返回值矩阵的指针
 */
		Matrix MatrixMultiply(Matrix &mat_1, Matrix &mat_2);

/* Function: AddElement
 * -------------------
 * 向矩阵添加元素
 */
		void AddElement(double element);

/* Function: GetRowNum
 * -------------------
 * 返回行
 */
		long GetRowNum();

/* Function: GetColNum
 * -------------------
 * 返回行
 */
		long GetColNum();

/* Function: GetElement
 * ---------------------
 * 返回输入的行列对应元素
 */
		double GetElement(long row, long col);

	private:
		//记录行列值
		long row_;
		long col_;
		//记录哪行已经填满
		long row_unfull_pos_;
		long col_unfull_pos_;
		//一列为一张图片的像素值，或者多张图片的像素值，一行为batchSize
		//假如表示权重，则对应该图片的权重值得到hidden层的图片
		double **all_element_;
};
