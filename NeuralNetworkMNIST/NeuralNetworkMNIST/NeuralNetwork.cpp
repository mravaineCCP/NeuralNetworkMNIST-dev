#include "NeuralNetwork.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>


using namespace std;

void NeuralNetwork::init_array()
{
	//Input Layer to Hidden layer initialization

	for (int i = 0; i < num_InputNeurons; ++i)
	{
		weight1[i] = new double[num_HiddenNeurons];
		delta1[i] = new double[num_HiddenNeurons];
	}

	out1 = new double[num_InputNeurons];

	//Hidden layer initialization
	for (int i = 0; i < num_HiddenNeurons; ++i)
	{
		weight2[i] = new double[num_OutputNeurons];
		delta2[i] = new double[num_OutputNeurons];
	}

	in2 = new double[num_HiddenNeurons];
	out2 = new double[num_HiddenNeurons];
	theta2 = new double[num_HiddenNeurons];

	//Output Layer Initialization

	in3 = new double[num_OutputNeurons];
	out3 = new double[num_OutputNeurons];
	theta3 = new double[num_OutputNeurons];

	//Initialization of random weight to prevent local optima
	for (int i = 0; i < num_InputNeurons; ++i)
	{
		for (int j = 0; j < num_HiddenNeurons; ++j)
		{
			int sign = rand() % 2;

			weight1[i][j] = (double)(rand() % 6) / 10.0;

			// Alternate weight sign

			if (sign == 1)
			{
				weight1[i][j] = -weight1[i][j];
			}
		}
	}

	for (int i = 0; i < num_HiddenNeurons; ++i)
	{
		for (int j = 0; j < num_OutputNeurons; ++j)
		{
			int sign = rand() % 2;

			weight2[i][j] = (double)(rand() % 10 + 1) / (10.0 * num_OutputNeurons);

			// Alternate weight sign

			if (sign == 1)
			{
				weight2[i][j] = -weight2[i][j];
			}
		}
	}

}

// Basic Sigmoid function for our outputs

double NeuralNetwork::Sigmoid(double input)
{
	return 1.0 / (1.0+exp(-input));
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
				image[i][j] = 0;
			}
			else
			{
				image[i][j] = 1;
			}
		}
	}

	// Initialize the output values of our input layer

	for (int j = 0; j < image_Height; ++j)
	{
		for (int i = 0; i < image_Width; ++i)
		{
			int pos = i+1 + j * image_Width;
			out1[pos] = image[i][j];
		}
	}

	//Read Label of image (0-9)

	for (int i = 0; i < num_OutputNeurons; ++i)
	{
		expectedValue[i] = 0.0;
	}

	expectedValue[label[image_idx]] = 1.0;

	cout << "Label: " << (int)label[image_idx] << endl;
}

//In the forward propgatation we take an image input and pass it from Input layer to Output layer and train our network.

void NeuralNetwork::ForwardPropagation()
{
	//Reset the input of our hidden layer

	for (int i = 0; i < num_HiddenNeurons; ++i)
	{
		in2[i] = 0.0;
	}

	//Reset the input of our Output layer

	for (int i = 0; i < num_OutputNeurons; ++i)
	{
		in3[i] = 0.0;
	}

	//Predict weight for every input of hidden layer

	for (int i = 0; i < num_InputNeurons; ++i)
	{
		for (int j = 0; j < num_HiddenNeurons; ++j)
		{
			in2[j] += out1[i] * weight1[i][j];
		}
	}

	//Predict output of hidden layer

	for (int i = 0; i < num_HiddenNeurons; ++i)
	{
		out2[i] = Sigmoid(in2[i]);
	}

	//Predict weight for every input of output layer

	for (int i = 0; i < num_HiddenNeurons; ++i)
	{
		for (int j = 0; j < num_OutputNeurons; ++j)
		{
			in3[j] += out2[i] * weight2[i][j];
		}
	}

	//Predict output of output layer

	for (int i = 0; i < num_OutputNeurons; ++i)
	{
		out3[i] = Sigmoid(in3[i]);
	}
}

//In the Backward propgatation we take our output and go back to the input to check if we get the same image. We then adapt the weight.

void NeuralNetwork::BackwardPropagation()
{
	double sum;

	for (int i = 0; i < num_OutputNeurons; ++i)
	{
		theta3[i] = out3[i] * (1 - out3[i]) * (expectedValue[i] - out3[i]);
		//theta3[i] = out3[i] - expectedValue[i];
	}

	for (int i = 0; i < num_HiddenNeurons; ++i)
	{
		sum = 0.0;

		for (int j = 0; j < num_OutputNeurons; ++j)
		{
			sum += weight2[i][j] * theta3[j];
		}

		theta2[i] = out2[i] * (1.0 - out2[i]) * sum;
	}

	for (int i = 0; i < num_HiddenNeurons; ++i)
	{
		for (int j = 0; j < num_OutputNeurons; ++j)
		{
			//delta2[i][j] = theta3[j] * out2[i];
			//delta2[i][j] = theta3[j] * out2[i] + delta2[i][j];
			delta2[i][j] = (learningRate * theta3[j] * out2[i]) + (momentum * delta2[i][j]);
			weight2[i][j] += delta2[i][j];
		}
	}

	for (int i = 0; i < num_InputNeurons; ++i)
	{
		for (int j = 0; j < num_HiddenNeurons; ++j)
		{

			//delta1[i][j] = theta2[j] * out1[i];
			//delta1[i][j] = theta2[j] * out1[i] + delta1[i][j];
			delta1[i][j] = (learningRate * theta2[j] * out1[i]) + (momentum * delta1[i][j]);
			weight1[i][j] += delta1[i][j];

		}
	}


}

// Basic function that loops the learning process of our neural network.

int NeuralNetwork::Network_Learning()
{
	for (int i = 0; i < num_InputNeurons; ++i)
	{
		for (int j = 0; j < num_HiddenNeurons; ++j)
		{
			delta1[i][j] = 0.0;

		}
	}

	for (int i = 0; i < num_HiddenNeurons; ++i)
	{
		for (int j = 0; j < num_OutputNeurons; ++j)
		{
			delta2[i][j] = 0.0;

		}
	}

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
	return num_Iterations;
}

//Return the error of the neural network compared to the real image value.

double NeuralNetwork::square_error()
{
	double result = 0.0;

	for (int i = 0; i < num_OutputNeurons; ++i)
	{
		result += (out3[i] - expectedValue[i]) * (out3[i] - expectedValue[i]);
	}

	result *= 0.5;

	return result;
}

//Save our neural network to a file that we will use to predict new images labels.

void NeuralNetwork::MatrixToFile(string file_name)
{
	ofstream file(file_name.c_str(), ios::out);

	for (int i = 0; i < num_InputNeurons; ++i)
	{
		for (int j = 0; j < num_HiddenNeurons; ++j)
		{
			file << weight1[i][j] << " ";
		}

		file << endl;
	}

	for (int i = 0; i < num_HiddenNeurons; ++i)
	{
		for (int j = 0; j < num_HiddenNeurons; ++j)
		{
			file << weight2[i][j] << " ";
		}

		file << endl;
	}

	file.close();
}

void NeuralNetwork::Load_NeuralNetwork_Mode(std::string file_name)
{
}

void NeuralNetwork::Prediction()
{
}

