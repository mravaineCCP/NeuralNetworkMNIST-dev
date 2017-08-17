#pragma once
#ifndef NEURALNETWORK_TRAIN_H
#define NEURALNETWORK_TRAIN_H

#include "NeuralNetwork.h"


class NeuralNetwork_Train : NeuralNetwork
{
public:

	void Load_NeuralNetwork_Model(std::string file_name);

	void Test_Prediction();

private :


};
#endif // !NEURALNETWORK_TRAIN_H