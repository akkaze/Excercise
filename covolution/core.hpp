#include <opencv2/opencv.hpp>

#include <cblas.h>

#include "im2col.hpp"

class Node
{
public:
	int nums_;
	int rows_;
	int cols_;
	int channels_;

	long size_;

	double* value_;

	std::vector<Node*> inputs_;

	Node(int nums,
		int channels,
		int rows,
		int cols)
	{
		nums_ = nums;
		channels_ = channels;
		rows_ = rows;
		cols_ = cols;
		size_ = nums *
			channels *
			rows *
			cols;
		value_ = new double[size_];
		memset(value_,
			0,
			sizeof(double) * size_);
	}


	~Node()
	{
		delete value_;
	}

};

class Data : public Node
{
public:
	void Data(const std::string& imgname)
	{
		cv::Mat im = cv::imread(imgname,
			cv::IMREAD_UNCHANGED);
		nums_ = 1;
		channels_ = im.channels();
		rows_ = im.rows;
		cols_ = im.cols;

		size_ = nums *
			channels *
			rows *
			cols;
		value_ = new double[size_];
		memset(value_,
			0,
			sizeof(double) * size_);

		for(int num = 0; num < nums_; num++)
			for(int row = 0; row < rows_; row++)
				for(int col = 0; col < cols_; col++)
					for(int channel = 0; channel < channels_; channel++)
							*(value_ + num * rows_ *
								cols_ * channels_ +
								row * cols_ * channels_ +
								col * channels_ +
								channel) = 
							*(im.data + row * cols_ * channels_ +
								col * channels_ +
								channel);
	}
};

class Param : public Node
{
public:
	bool fixed_;

	Param(size_t nums,
		size_t channels,
		size_t rows,
		size_t cols)
	: Node(nums,channels,rows,cols)
	{
		fixed_ = false;

	}

	using Node::~Node;
};

class Conv : public Node
{
public:
	size_t window_;
	size_t stride_;
	size_t padding_;

	Param* W_;
	double* imcol_;

	Conv(Node* input, size_t num_filters,
		size_t window = 5,size_t stride = 1)
	{
		inputs_.push_back(input);
		window_ = window;
		stride_ = stride;
		nums_ = input->nums_;
		channels_ = num_filters;

		padding_ = (window - 1) / 2;
		rows_ = (input->rows_ + 2 * padding_ - window) / stride + 1;
		cols_ = (input->cols_ + 2 * padding_ - window) / stride + 1;

		size_ = nums_ *
			channels_ *
			rows_ *
			cols_;
		W_ = new Param(input->channels_,
			window,
			window,
			num_filters);
		

		value_ = new double[size_];
		memset(value_,
			0,
			sizeof(double) * size_);
	}

	~Conv()
	{
		delete[] value_;
		delete[] imcol_;
		delete W_;
	}
	void forward()
	{
		im2col(value_,
			imcol_,
			nums_,
			channels_,
			rows_,
			cols_,
			window_,
			window_,
			stride_);

		int  m = 
	}
};

class View
{
public:
	View(Node* input,const std::string& imgname)
	{
		cv::Mat im(input->rows_,
			input->cols_,
			CV_8UC3);
		for(int num = 0; num < nums_; num++)
			for(int row = 0; row < rows_; row++)
				for(int col = 0; col < cols_; col++)
					for(int channel = 0; channel < channels_; channel++)
					{
						double* ptr = input->value_ + num * rows_ *
							cols_ * channels_ +
							row * cols_ * channels_ +
							col * channels_ +
							channel;
						uchar* im_prt = im.data + row * cols_ * channels_ +
							col * channels_ +
							channel;
						if(*ptr >= 0 && *ptr < 255)
							*im_ptr = *ptr;
						else if(*ptr < 0)
							*im_ptr = 0;
						else
							*im_ptr = 255;
					}
		cv::imsave(imgname,im);
	}
};