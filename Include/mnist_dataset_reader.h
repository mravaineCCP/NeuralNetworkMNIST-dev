#pragma once

#include <iostream>
#include <fstream>

#include "opencv2/core.hpp"    // for cv::Mat
#include "opencv2/highgui.hpp" // for CV_RGB
#include "opencv2/imgproc.hpp" // for cv::putText

using namespace std;
using namespace cv;

class mnist_dataset_reader
{
public:

	mnist_dataset_reader(string path_to_extracted_mnist_files); // reads in all training / test images + labels

	unsigned char**         get_train_images();

	unsigned char**         get_test_images();

	unsigned char*          get_train_labels();

	unsigned char*          get_test_labels();

	Mat*                    get_mnist_image_as_cvmat(unsigned char** images, int image_idx);

	Mat*                    get_board_of_sample_images(unsigned char** images, unsigned char* labels, int nr_of_images);





private:

	unsigned char**         mnist_dataset_reader::read_mnist_images(string full_path, int& number_of_images, int& image_size); // for reading in images

	unsigned char*          mnist_dataset_reader::read_mnist_labels(string full_path, int& number_of_labels);                  // for reading in ground truth image labels


	string                  path_to_extracted_mnist_files; // where you have extracted the files 

	unsigned char**         train_images;                  // all the 60.000 training images of size 28x28 

	unsigned char**         test_images;                   // all the 10.000 test     images of size 28x28

	unsigned  char*         train_labels;                  // all the 60.000 ground truth labels for the training images

	unsigned  char*         test_labels;                   // all the 10.000 ground truth labels for the test     images

	int                     nr_train_images_read;          // should be 60.000

	int                     nr_test_images_read;           // should be 10.000

};
