#pragma once
#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <string>
#include <Eigen/Dense>

/*Setup for the Neural network
* The setup for this neural network is based on 3 Layer : Input Layer - 1 Hidden layer - Output Layer
*/

class NeuralNetwork
{
public :

	
	void init_array();

	Eigen::VectorXf Sigmoid(const Eigen::VectorXf& input);

	void  ForwardPropagation();

	void BackwardPropagation();

	float square_error();

	float accuracy(int correct_predictions, int test_iterations);

	void inputNetwork(unsigned char** images, unsigned char* label, int image_idx);

	int Network_Learning();

	void MatrixToFile(const std::string& file_name);

	void Load_NeuralNetwork_Model(const std::string& file_name);

	int Prediction();

private :


		//Number of input = Image Height * Width. (28 * 28 = 784)

	const static int num_InputNeurons = 784;

		//Number of neurons in the Hidden Layer.

	const static int num_HiddenNeurons = 128;

		//Number of output (MNIST have 0-9 digit = 10 Possible output)

	const static int num_OutputNeurons = 10;

		//Tweak this settings depending on the error result of the Learning process

	const float learningRate = 0.001f;

		//Momentum of Delta of Backward propagation

	const float momentum = 0.9f;

		//Number of Iteration from the neural network on a specific training example

	const static int num_Iterations = 512; //512

		//Error minimum

	const float epsilon = 0.001f;

		//Input Layer to Hidden Layer

	Eigen::MatrixXf weight1;
	Eigen::MatrixXf delta1;

	Eigen::VectorXf out1;

		//Hidden Layer to Output Layer

	Eigen::MatrixXf weight2;
	Eigen::MatrixXf delta2;

	Eigen::VectorXf  in2;
	Eigen::VectorXf  out2;
	Eigen::VectorXf  theta2;

		//Output layer

	Eigen::VectorXf expectedValue;

	Eigen::VectorXf in3;
	Eigen::VectorXf out3;
	Eigen::VectorXf theta3;


		//Image 28x28 GrayScale value

	const static int image_Width = 28;
	const static int image_Height = 28;

		//Bit Data of image

	Eigen::MatrixXf image;
	int image_Label = 0;

};
#endif // !NEURALNETWORK_H
