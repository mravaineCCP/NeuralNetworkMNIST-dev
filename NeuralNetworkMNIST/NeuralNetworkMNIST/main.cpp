#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <conio.h>  
#include <chrono>
#include <ratio>
#include <ctime>

typedef std::chrono::steady_clock Clock;

#include "mnist_dataset_reader.h"
#include "NeuralNetwork.h"


using namespace cv;
using namespace std;

int main()
{
	int choice = 5;
	string path_to_extracted_mnist_files = "MNIST";										//Path to read images
	string path_to_matrix_save = "Data\\Matrix.txt";			//Path to save our NN matrix.
	mnist_dataset_reader my_reader(path_to_extracted_mnist_files);
	NeuralNetwork network;

	int prediction = 0;
	float accuracy = 0.0;

	Eigen::initParallel();
	Eigen::setNbThreads(8);
	cout << "Eigen threads: " << Eigen::nbThreads() << endl;

	Clock::time_point startTimer = Clock::now();
	Clock::time_point endTimer = Clock::now();
	double timeSpent = 0.0;

	/*
	Mat* img = my_reader.get_mnist_image_as_cvmat(my_reader.get_train_images(), sample);
	Mat img_resized;
	resize(*img, img_resized, cv::Size(0, 0), 4, 4);
	imshow("A sample digit", img_resized);
	*/


	//NEURAL NETWORK
	while (choice != 0)
	{

	cout << "1 : Train model" << endl << "2 : Test model (if already trained)" << endl << "0 : Quit" << endl;
	cin >> choice;


		switch (choice)
		{

		case 1:

			//Loop in the 60000 example of the traning set images and train our network.

			startTimer = Clock::now();

			network.init_array();

			for (int sample = 1; sample <= 100; ++sample)
			{

				network.inputNetwork(my_reader.get_train_images(), my_reader.get_train_labels(), sample);

				int nIterations = network.Network_Learning();

				// Write down the squared error
				cout << "Sample: " << sample << endl;
				cout << "No. iterations: " << nIterations << endl;
				cout << "Error : " << network.square_error() << endl;

				//Save intermediate network
				//if (sample % 100 == 0)
				//{
				//	cout << "Saving the network to :" << path_to_matrix_save << " file." << endl;
				//	network.MatrixToFile(path_to_matrix_save);
				//}
			}

			//Save final network
			network.MatrixToFile(path_to_matrix_save);

			endTimer = Clock::now();
			
			timeSpent = (endTimer - startTimer).count()  * ((double)Clock::period::num / Clock::period::den);;
			

			cout << "Time spent : " << timeSpent << " secondes" << endl;

			break;

		case 2:

			//prediction = 0;
			network.init_array();
			network.Load_NeuralNetwork_Model(path_to_matrix_save);


			for (int sample = 0; sample < 1000; ++sample)
			{
				network.inputNetwork(my_reader.get_test_images(), my_reader.get_test_labels(), sample);

				network.ForwardPropagation();
				prediction += network.Prediction();
			}

			cout << endl << "Number of good prediction : " << prediction << endl << endl;
			accuracy = network.accuracy(prediction, 9000);
			cout << "Accuracy : " << accuracy << "%" << endl <<  endl;

			break;		

		default:

			cout << "Enter a good choice" << endl;
			break;


		}

	}







	
	/*
	Mat* samples_as_image = my_reader.get_board_of_sample_images(my_reader.get_train_images(), my_reader.get_train_labels(), 60000);
	imshow("Some sample MNIST training images", *samples_as_image);


	//waitKey(0);

	//_getch();
	*/

	return 0;
}