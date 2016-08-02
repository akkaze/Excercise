#include "image_generator.hpp"

#include "google/gflags.h"

#include <vector>
#include <iostream>

DEFINE_string(operation,"noise","");
DEFINE_string(read_from_dir,"/home/zak/input/","");
DEFINE_string(save_to_dir,"/home/zak/preview/","");
DEFINE_string(ext,"jpg","");
DEFINE_int32(nums,20,"");

DEFINE_double(rotation_range,0.2,"");
DEFINE_double(height_shift_range,0.2,"");
DEFINE_double(width_shift_range,0.2,"");
DEFINE_double(shear_range,0.2,"");
DEFINE_double(low_zoom_factor,0.8,"");
DEFINE_double(high_zoom_factor,1.2,"");
DEFINE_bool(horizontal_flip,true,"");
DEFINE_bool(vertical_flip,true,"");

DEFINE_string(noise_type,"pepper_salt","");
DEFINE_double(density,0.01,"");
DEFINE_double(mean,100,"");
DEFINE_double(stddev,5,"");

DEFINE_double(alpha_range,0.2,"");
DEFINE_double(beta,5,"");

DEFINE_int32(small_ksize,3,"");
DEFINE_int32(big_ksize,7,"");
DEFINE_double(low_sigma,1.0,"");
DEFINE_double(high_sigma,2.2,"");

int main(int argc,char** argv)
{	
	google::ParseCommandLineFlags(&argc, &argv, true);
	if(FLAGS_operation == "geometric")
	{
		ImageDataGenerator::OpType type = ImageDataGenerator::GEOMETRIC;
		ImageDataGenerator image_data_generator(FLAGS_read_from_dir,FLAGS_save_to_dir,FLAGS_nums,FLAGS_ext,
FLAGS_rotation_range,FLAGS_height_shift_range,FLAGS_width_shift_range,
FLAGS_shear_range,FLAGS_low_zoom_factor,FLAGS_high_zoom_factor,FLAGS_horizontal_flip,FLAGS_vertical_flip);
		image_data_generator.apply(type);
	}	
	else if(FLAGS_operation == "noise")
	{
		ImageDataGenerator::OpType type = ImageDataGenerator::NOISE;
		ImageDataGenerator image_data_generator(FLAGS_read_from_dir,FLAGS_save_to_dir,FLAGS_nums,
FLAGS_ext,FLAGS_noise_type,FLAGS_density,FLAGS_mean,FLAGS_stddev);
		image_data_generator.apply(type);
	}
	else if(FLAGS_operation == "contrast")
	{
		ImageDataGenerator::OpType type = ImageDataGenerator::CONTRAST;
		ImageDataGenerator image_data_generator(FLAGS_read_from_dir,FLAGS_save_to_dir,FLAGS_nums,
FLAGS_ext,FLAGS_alpha_range,FLAGS_beta);
		image_data_generator.apply(type);
	}
	else if(FLAGS_operation == "blur")
	{
		ImageDataGenerator::OpType type = ImageDataGenerator::BLUR;
		ImageDataGenerator image_data_generator(FLAGS_read_from_dir,FLAGS_save_to_dir,FLAGS_nums,
FLAGS_ext,FLAGS_small_ksize,FLAGS_big_ksize,FLAGS_low_sigma,FLAGS_high_sigma);
		image_data_generator.apply(type);
	}
	return 0;
}
