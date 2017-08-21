#include "NeuralNetwork.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <omp.h> 

using namespace std;
using namespace Eigen;

void NeuralNetwork::init_array()
{
	////Input Layer to Hidden layer initialization

	weight1.resize(num_InputNeurons,num_HiddenNeurons);
	delta1.resize(num_InputNeurons, num_HiddenNeurons);

	out1.resize(num_InputNeurons);
	out1 = VectorXf::Zero(num_HiddenNeurons);
	//Hidden layer initialization

	weight2.resize(num_HiddenNeurons, num_OutputNeurons);
	delta2.resize(num_HiddenNeurons, num_OutputNeurons);

	in2.resize(num_HiddenNeurons);
	out2.resize(num_HiddenNeurons);
	theta2.resize(num_HiddenNeurons);

	in2 = VectorXf::Zero(num_HiddenNeurons);
	out2 = VectorXf::Zero(num_HiddenNeurons);
	theta2 = VectorXf::Zero(num_HiddenNeurons);

	//Output Layer Initialization

	in3.resize(num_OutputNeurons);
	out3.resize(num_OutputNeurons);
	theta3.resize(num_OutputNeurons);

	in3 = VectorXf::Zero(num_HiddenNeurons);
	out3 = VectorXf::Zero(num_HiddenNeurons);
	theta3 = VectorXf::Zero(num_HiddenNeurons);
	//Image initialization

	image.resize(image_Height, image_Width);
	image = MatrixXf::Zero(image_Height, image_Width);

	//Initialization of random weight to prevent local optima

	weight1.setRandom();
	weight2.setRandom();




}

// Basic Sigmoid function for our outputs

VectorXf NeuralNetwork::Sigmoid(const VectorXf& input)
{
	VectorXf inputNeg = -1.0f * input;
	VectorXf output(input.size());


	output = (1.0f + exp(inputNeg.array())).cwiseInverse();

	//cout << output << endl;

	return output;
}

// This function gets the intensity of each pixel of an image and transform it into a grayscale bit image that we can use as input of our neural network.

void NeuralNetwork::inputNetwork(unsigned char** images, unsigned char* label, int image_idx)
{
	int idx_1d = 0;

	unsigned char gray_value = 0;

	//Read image data and transform to bit image


	for (int j = 0; j < image_Height; ++j)
	{
		for (int i = 0; i < image_Width; ++i)
		{
			gray_value = images[image_idx][idx_1d++];

			if (gray_value == 0)
			{
				image(i, j) = 0.f;
			}
			else
			{
				image(i, j) = 1.f;
			}
		}
	}

	// Initialize the output values of our input layer

	Map<VectorXf> image_vector(image.data(), image.size());
	out1 = image_vector;	

	//Read Label of image (0-9)

	expectedValue = VectorXf::Zero(num_OutputNeurons);
	expectedValue(label[image_idx]) = 1;


	image_Label = (int)label[image_idx];
	cout << "Label: " << image_Label << endl;
}

//In the forward propgatation we take an image input and pass it from Input layer to Output layer and train our network.

void NeuralNetwork::ForwardPropagation()
{
	//Predict weight for every input of hidden layer

	in2 = weight1.transpose() * out1;

	//Predict output of hidden layerZ

	out2 = Sigmoid(in2);

	//Predict weight for every input of output layer


	in3 = weight2.transpose() * out2;


	//Predict output of output layer

	out3 = Sigmoid(in3);


}

//In the Backward propgatation we take our output and go back to the input to check if we get the same image. We then adapt the weight.

void NeuralNetwork::BackwardPropagation()
{
	VectorXf Out_OneMatrix = VectorXf::Ones(num_OutputNeurons);
	
	theta3 = (out3.cwiseProduct((Out_OneMatrix - out3))).cwiseProduct((expectedValue - out3));
	/*theta3[i] = out3[i] * (1 - out3[i]) * (expectedValue[i] - out3[i]);*/

	VectorXf Hidden_OneMatrix = VectorXf::Ones(num_HiddenNeurons);

	theta2 = (weight2 * theta3).cwiseProduct((out2.cwiseProduct((Hidden_OneMatrix - out2))));

	delta2 = learningRate * (out2 * theta3.transpose()) + (momentum*delta2);
	weight2 += delta2;

	delta1 = learningRate * (out1 * theta2.transpose()) + (momentum*delta1);
	weight1 += delta1;


}

// Basic function that loops the learning process of our neural network.

int NeuralNetwork::Network_Learning()
{
	//memset(delta1, 0, sizeof(float)* num_InputNeurons * num_HiddenNeurons);
	//memset(delta2, 0, sizeof(float)* num_HiddenNeurons * num_OutputNeurons);


	delta1 = MatrixXf::Zero(num_InputNeurons, num_HiddenNeurons);
	delta2 = MatrixXf::Zero(num_HiddenNeurons, num_OutputNeurons);
		  

	for (int i = 0; i < num_Iterations; ++i)
	{
		//Loop inside our neural network

		ForwardPropagation();
		BackwardPropagation();

		//If Square_error is under the maximum error we want, no need to keep going.

		if (square_error() < epsilon)
		{
			return i;
		}
	}

	//cout << "Label: " << image_Label << endl << "Output:" << endl << out3 << endl << endl;

	return num_Iterations;
}

//Return the error of the neural network compared to the real image value.

float NeuralNetwork::square_error()
{
	float result = 0.f;

	for (int i = 0; i < out3.size(); ++i)
	{
		result += (out3(i) - expectedValue(i)) * (out3(i) - expectedValue(i));
	}

	result *= 0.5f;

	return result;
}

//Save our neural network to a file that we will use to predict new images labels.

void NeuralNetwork::MatrixToFile(const string& file_name)
{
	ofstream file(file_name.c_str(), ios::out);


	for (int i = 0; i < num_InputNeurons; ++i)
	{
		for (int j = 0; j < num_HiddenNeurons; ++j)
		{
			file << weight1(i,j) << " ";
		}

		file << endl;
	}


	for (int i = 0; i < num_HiddenNeurons; ++i)
	{
		for (int j = 0; j < num_OutputNeurons; ++j)
		{
			file << weight2(i,j) << " ";
		}

		file << endl;
	}

	file.close();
}

void NeuralNetwork::Load_NeuralNetwork_Model(const string& file_name)
{
	ifstream file(file_name.c_str(), ios::in);

		for (int i = 0; i < num_InputNeurons; ++i)
		{
			for (int j = 0; j < num_HiddenNeurons; ++j)
			{
				file >> weight1(i,j);
			}
		}

		for (int i = 0; i < num_HiddenNeurons; ++i)
		{
			for (int j = 0; j < num_OutputNeurons; ++j)
			{
				file >> weight2(i,j);
			}
		}

		file.close();
}

int NeuralNetwork::Prediction()
{
	int predict = 0;
	int num_Correct = 0;

	for (int i = 1; i < num_OutputNeurons; ++i)
	{
		if (out3[i] > out3[predict])
		{
			predict = i;
		}
	}

	if (predict == image_Label)
	{
		++num_Correct;
		cout << "Classification: YES. Label = " << image_Label << ". Predict = " << predict << endl << endl;
	}
	else
	{
		cout << "Classification: NO.  Label = " << image_Label << ". Predict = " << predict << endl;
	}

	return num_Correct;
}

float NeuralNetwork::accuracy(int correct_predictions, int test_iterations)
{
	return ((float)correct_predictions / (float)test_iterations) * 100.0f;
}

