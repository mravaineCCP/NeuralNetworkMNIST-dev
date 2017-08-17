#pragma once
#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <string>

/*Setup for the Neural network
* The setup for this neural network is based on 3 Layer : Input Layer - 1 Hidden layer - Output Layer
*/

class NeuralNetwork
{
public :

	
	void init_array();

	double Sigmoid(double input);

	void  ForwardPropagation();

	void BackwardPropagation();

	double square_error();

	void inputNetwork(unsigned char** images, unsigned char* label, int image_idx);

	int Network_Learning();

	void MatrixToFile(std::string file_name);

	void Load_NeuralNetwork_Mode(std::string file_name);

	void Prediction();

private :


		//Number of input = Image Height * Width. (28 * 28 = 784)

	const static int num_InputNeurons = 784;

		//Number of neurons in the Hidden Layer.

	const static int num_HiddenNeurons = 128;

		//Number of output (MNIST have 0-9 digit = 10 Possible output)

	const static int num_OutputNeurons = 10;

		//Tweak this settings depending on the error result of the Learning process

	const double learningRate = 0.001;

		//Momentum of Delta of Backward propagation

	const double momentum = 0.9;

		//Number of Iteration from the neural network on a specific training example

	const static int num_Iterations = 512;

		//Error minimum

	const double epsilon = 0.001;

		//Input Layer to Hidden Layer

	double *weight1[num_InputNeurons], *delta1[num_InputNeurons], *out1;

		//Hidden Layer to Output Layer

	double *weight2[num_HiddenNeurons], *in2, *delta2[num_HiddenNeurons], *out2, *theta2;;

		//Output layer

	double *in3, *out3, *theta3;
	double expectedValue[num_OutputNeurons];

		//Image 28x28 GrayScale value

	const static int image_Width = 28;
	const static int image_Height = 28;

		//Bit Data of image

	int image[image_Width][image_Height];


};
#endif // !NEURALNETWORK_H
