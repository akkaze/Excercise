#include "opencv2/opencv.hpp"

#include <cassert>
#include <random>
#include <cmath>
#include <vector>
#include <unordered_map>

#include <iostream>

#include <boost/algorithm/string/split.hpp>
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

class ImageDataGenerator
{
public:
enum NoiseType
{
	
};

enum BlurType
{
	GAUSSIAN = 1L << 0,
	MEDIAN = 1L << 1
};

struct BlurParam
{
	BlurParam(BlurType blur_type,int ksize,double sigma) :
		blur_type_(blur_type),ksize_(ksize),sigma_(sigma) {}
	BlurType blur_type_;
	int ksize_;
	double sigma_;
};

enum OpType
{
	GEOMETRIC = 1L << 0,
	NOISE = 1L << 1,
	CONTRAST = 1L << 2,
	BLUR = 1L << 3
};

public:
	ImageDataGenerator(std::string read_from_dir,
	std::string save_to_dir,
	int num,
	std::string ext,
	double rotation_range,
	double height_shift_range,
	double width_shift_range,
	double shear_range,
	double low_zoom_range,
	double high_zoom_range,
	bool horizontal_flip,
	bool vertical_flip) :
	rotation_range_(rotation_range),
	height_shift_range_(height_shift_range),
	width_shift_range_(width_shift_range),
	shear_range_(shear_range),
	low_zoom_range_(low_zoom_range),
	high_zoom_range_(high_zoom_range),
	horizontal_flip_(horizontal_flip),
	vertical_flip_(vertical_flip),
	read_from_dir_(read_from_dir),
	save_to_dir_(save_to_dir),
	num_(num),
	ext_(ext) {}

	ImageDataGenerator(std::string read_from_dir,
	std::string save_to_dir,
	int num,
	std::string ext,
	std::string noise_type,
	double density,
	double mean,
	double stddev
	) :
	read_from_dir_(read_from_dir),
	save_to_dir_(save_to_dir),
	num_(num),
	ext_(ext),
	noise_type_(noise_type),
	density_(density),
	mean_(mean),
	stddev_(stddev)
	{}	

	ImageDataGenerator(std::string read_from_dir,
	std::string save_to_dir,
	int num,
	std::string ext,
	double alpha_range,
	double beta
	) :
	read_from_dir_(read_from_dir),
	save_to_dir_(save_to_dir),
	num_(num),
	ext_(ext),
	alpha_range_(alpha_range),
	beta_(beta)
	{}	

	ImageDataGenerator(std::string read_from_dir,
	std::string save_to_dir,
	int num,
	std::string ext,
	int small_ksize,
	int big_ksize,
	double low_sigma,
	double high_sigma
	) :
	read_from_dir_(read_from_dir),
	save_to_dir_(save_to_dir),
	num_(num),
	ext_(ext),
	small_ksize_(small_ksize),
	big_ksize_(big_ksize),
	low_sigma_(low_sigma),
	high_sigma_(high_sigma)
	{}	

	cv::Mat random_transform(cv::Mat src)
	{
		std::random_device rd;
		std::mt19937 eng(rd());
		//rotation matrix
		std::uniform_real_distribution<double> ang_dis(-rotation_range_,rotation_range_);
		double theta = CV_PI / 180 	* ang_dis(eng);
		cv::Mat rotation_matrix = (cv::Mat_<double>(3,3) << std::cos(theta),-std::sin(theta),0,std::sin(theta),std::cos(theta),0,0,0,1);
		
		//translate matrix
		std::uniform_real_distribution<double> height_dis(-height_shift_range_,height_shift_range_);
		double tx = height_dis(eng) * src.rows;
		
		std::uniform_real_distribution<double> width_dis(-width_shift_range_,width_shift_range_);
		double ty = width_dis(eng) * src.cols;
		
		cv::Mat translation_matrix = (cv::Mat_<double>(3,3) << 1,0,tx,0,1,ty,0,0,1);
	
		//shear matrix
		std::uniform_real_distribution<double> shear_dis(-shear_range_,shear_range_);
		double shear = shear_dis(eng);
		cv::Mat shear_matrix = (cv::Mat_<double>(3,3) << 1,-std::sin(shear),0,0,std::cos(shear),0,0,0,1);
		
		//scale matrix
		std::uniform_real_distribution<double> zoom_dis(low_zoom_range_,high_zoom_range_);
		double zx = zoom_dis(eng);
		double zy = zoom_dis(eng);
		cv::Mat zoom_matrix = (cv::Mat_<double>(3,3) << zx,0,0,0,zy,0,0,0,1); 
		
		//composition matrix
		cv::Mat transform_matrix = rotation_matrix * translation_matrix * shear_matrix * zoom_matrix;
		cv::Mat trans_mat = transform_matrix(cv::Rect(0,0,3,2)).clone();
		//apply transformation
		cv::Mat dst;
		cv::warpAffine(src,dst,trans_mat,cv::Size(src.rows,src.cols),cv::INTER_LINEAR,cv::BORDER_CONSTANT);
		
		//filp image		
		if(vertical_flip_)
		{
			std::uniform_real_distribution<double> vert_dis(0,1);
			double rand = vert_dis(eng);
			if(rand < 0.5)
				cv::flip(dst,dst,1);
		}
		if(horizontal_flip_)
		{
			std::uniform_real_distribution<double> hori_dis(0,1);
			double rand = hori_dis(eng);
			if(rand < 0.5)
				cv::flip(dst,dst,0);
		}
		return dst;
	}

	void apply(const OpType& type)
	{
		fs::path read_from_dir_path(read_from_dir_);
		fs::directory_iterator end;
		for(fs::directory_iterator iter(read_from_dir_path); iter != end; ++iter)
		{
			std::string read_from_imgname = iter->path().string();
			std::string basename = fs::basename(iter->path());
		
			cv::Mat src = cv::imread(read_from_imgname,cv::IMREAD_UNCHANGED);
			for(size_t idx = 0; idx < num_; idx++)
			{
				std::string new_imgname = save_to_dir_ + basename + std::to_string(idx) + '.' + ext_;
				cv::Mat new_mat;
				if(type == GEOMETRIC)
					new_mat = random_transform(src);
				else if(type == NOISE)
					new_mat = add_noise(src);
				else if(type == CONTRAST)
					new_mat = adjust_contrast(src);
				else if(type == BLUR)
					new_mat = blur(src);
				cv::imwrite(new_imgname,new_mat);
			}
		}
	}

cv::Mat add_noise(cv::Mat& src)
{
	int rows = src.rows;
	int cols = src.cols;
	int channels = src.channels();

	cv::Mat dst = src.clone();
	std::random_device rd;
	std::mt19937 gen(rd());
	
	std::uniform_int_distribution<unsigned int> row_dis(0,rows - 1);
	std::uniform_int_distribution<unsigned int> col_dis(0,cols - 1);
	
	std::uniform_real_distribution<double> mean_dis(mean_ - 10,mean_ + 10);
	std::uniform_real_distribution<double> stddev_dis(0,stddev_);
	std::uniform_real_distribution<double> density_dis(0,density_);
	
	double mean = mean_dis(gen);
	double stddev = stddev_dis(gen);
	double dens = density_dis(gen);

	
	int num_noises = rows * cols * dens;

	if(noise_type_ == "gaussian")
	{	
		std::normal_distribution<double> dis(mean_,stddev_);
		for(int i = 0; i < num_noises; i++)
		{
			unsigned int row = row_dis(gen);
			unsigned int col = col_dis(gen);
		
			for(int c = 0; c < channels; c++)
			{
				double noise = dis(gen);
					
				*(dst.data + row * cols * channels + col * channels + c) = cv::saturate_cast<uchar>(noise);		
			} 
		}	
	}
	else if(noise_type_ == "pepper_salt")
	{
		std::uniform_real_distribution<double> dis(0,1);
		for(int i = 0; i < num_noises; i++)
		{
			unsigned int row = row_dis(gen);
			unsigned int col = col_dis(gen);
			
			double noise = dis(gen);
			if(noise > 0.5)	
			{		
				for(int c = 0; c < channels; c++)
				{
						*(dst.data + row * cols * channels + col * channels + c) = 255;		
				
				}
			}
			else
			{		
				for(int c = 0; c < channels; c++)
				{
						*(dst.data + row * cols * channels + col * channels + c) = 0;		
				
				}
			}
		}
	}
	return dst;
}


cv::Mat adjust_contrast(cv::Mat& src)
{
	int rows = src.rows;
	int cols = src.cols;
	int channels = src.channels();

	cv::Mat dst = src.clone();
	std::random_device rd;
	std::mt19937 gen(rd());

	std::uniform_real_distribution<double> alpha_dis(1-alpha_range_,1+alpha_range_);
	std::uniform_real_distribution<double> beta_dis(-beta_,beta_);

	double alpha = alpha_dis(gen);
	double beta = beta_dis(gen);
	for(int row = 0; row < rows; row++)
	{
		for(int col = 0; col < cols; col++)
		{	
			for(int ch = 0; ch < channels; ch++)
			{
				uchar* data_ptr = dst.data + row * cols * channels + col * channels + ch;		
				*data_ptr = cv::saturate_cast<uchar>(alpha * *data_ptr + beta);		
			}
		} 
	}
	return dst;
}

cv::Mat blur(cv::Mat& src)
{
	cv::Mat dst;
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> ksize_dis(small_ksize_,big_ksize_);
	int ksize = 0;
	do
	{
		ksize = ksize_dis(gen);
	}while(ksize % 2 == 0);
	std::uniform_real_distribution<double> sigma_dis(low_sigma_,high_sigma_);
	double sigma = sigma_dis(gen);
	cv::GaussianBlur(src,dst,cv::Size(ksize,ksize),sigma,0);
	return dst;
}

private:
	//geometric params
	double rotation_range_;
	double height_shift_range_;
	double width_shift_range_;
	double shear_range_;
	double low_zoom_range_;
	double high_zoom_range_;
	bool horizontal_flip_;
	bool vertical_flip_;
	
	//noise params	
	std::string noise_type_;
	double density_;
	double mean_;
	double stddev_;

	//contrast params
	double alpha_range_;
	double beta_;

	//blur params
	int small_ksize_;
	int big_ksize_;
	double low_sigma_;
	double high_sigma_;
	//general params	
	std::string read_from_dir_;
	std::string save_to_dir_;
	int num_;
	std::string ext_;
};
