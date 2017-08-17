#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <conio.h>  


#include "mnist_dataset_reader.h"
#include "NeuralNetwork.h"


using namespace cv;
using namespace std;

int main()
{

	string path_to_extracted_mnist_files = "C:\\Users\\marc.se\\Documents\\Visual Studio 2015\\Projects\\MNIST_Reader\\MNIST";										//Path to read images
	string path_to_matrix_save = "C:\\Users\\marc.se\\Documents\\Visual Studio 2015\\Projects\\NeuralNetworkMNIST\\NeuralNetworkMNIST\\Data\\Matrix.txt";			//Path to save our NN matrix.
	mnist_dataset_reader my_reader(path_to_extracted_mnist_files);
	NeuralNetwork network;

	/*
	Mat* img = my_reader.get_mnist_image_as_cvmat(my_reader.get_train_images(), sample);
	Mat img_resized;
	resize(*img, img_resized, cv::Size(0, 0), 4, 4);
	imshow("A sample digit", img_resized);
	*/


	//NEURAL NETWORK

	network.init_array();

	//Loop in the 60000 example of the traning set images and train our network.

	for (int sample = 1; sample < 60000; ++sample)																													
	{

		network.inputNetwork(my_reader.get_train_images(), my_reader.get_train_labels(), sample);

		int nIterations = network.Network_Learning();

		// Write down the squared error
		cout << "Sample: " << sample << endl;
		cout << "No. iterations: " << nIterations << endl;
		cout << "Error : " << network.square_error() << endl;

		//Save intermediate network
		if (sample % 100 == 0)
		{
			cout << "Saving the network to :" << path_to_matrix_save << " file." << endl;
			network.MatrixToFile(path_to_matrix_save);
		}
	}


	//Save final network
	network.MatrixToFile(path_to_matrix_save);

	
	/*
	Mat* samples_as_image = my_reader.get_board_of_sample_images(my_reader.get_train_images(), my_reader.get_train_labels(), 60000);
	imshow("Some sample MNIST training images", *samples_as_image);


	//waitKey(0);

	//_getch();
	*/

	return 0;
}