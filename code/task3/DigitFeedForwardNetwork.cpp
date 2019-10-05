#include "DigitFeedForwardNetwork.h"
#include <iostream>
#include <random>
#include <iomanip>      // std::setprecision

void DigitFeedForwardNetwork::initialize(int seed)
{
	srand(seed);

	layerWeights.resize(numHiddenLayers+1); //num of layers with weights

	//input to first hidden layer
	layerWeights[0].resize(inputLayerSize);
	for (size_t i = 0; i < inputLayerSize; i++)
	{
		layerWeights[0][i].resize(hiddenLayerSize);
		for (size_t j = 0; j < hiddenLayerSize; j++)
		{
			layerWeights[0][i][j] = (rand() % 101 - 50) * 1.0 / 100; 	// This network cannot learn if the initial weights are set to zero.
		}
	}


cout << "1";
	//weights between hiddenlayers
	for(size_t i = 1; i < numHiddenLayers; i++){
		layerWeights[i].resize(hiddenLayerSize);
		for(size_t j = 0; j < hiddenLayerSize; j = 0){
			layerWeights[i][j].resize(hiddenLayerSize);
			for (size_t k = 0; k < hiddenLayerSize; k++)
			{
				layerWeights[i][j][k] = (rand() % 101 - 50) * 1.0 / 100; 	// This network cannot learn if the initial weights are set to zero.
			}

		}
	}
cout << "2";

	//weight from last hl to output layer
	layerWeights[numHiddenLayers].resize(hiddenLayerSize);
	for (size_t i = 0; i < hiddenLayerSize; i++)
	{
		layerWeights[numHiddenLayers][i].resize(outputSize);
		for (size_t j = 0; j < outputSize; j++)
		{
			layerWeights[numHiddenLayers][i][j] = (rand() % 101 - 50) * 1.0 / 100; 	// This network cannot learn if the initial weights are set to zero.
		}
	}
	cout <<"3";
}

void DigitFeedForwardNetwork::train(const vector< vector< double > >& x,
	const vector< double >& y, size_t numEpochs)
{
	size_t trainingexamples = x.size();


	// train the network
	for (size_t epoch = 0; epoch < numEpochs; epoch++)
	{
		// print
		cout << "epoch = " << epoch << ", outputs =";
		for (size_t example = 0; example < trainingexamples; example++)
		{
			// propagate the inputs forward to compute the outputs 
			vector< double > activationInput(inputLayerSize); // We store the activation of each node (over all input and hidden layers) as we need that data during back propagation.			
			for (size_t inputNode = 0; inputNode < inputLayerSize; inputNode++) // initialize input layer with training data
			{
				activationInput[inputNode] = x[example][inputNode];
			}
			
			vector< vector< double > > activationHidden; //[layer][node] = input
			activationHidden.resize(numHiddenLayers);
			vector< double > inputHidden(hiddenLayerSize);
			// calculate activations of hidden layers (for now, just one hidden layer)
			
			//input layer to first hidden layer
			activationHidden[0].resize(hiddenLayerSize);
			for (size_t hiddenNode = 0; hiddenNode < hiddenLayerSize; hiddenNode++)
			{
				double inputToHidden = 0;
				for (size_t inputNode = 0; inputNode < inputLayerSize; inputNode++)
				{
					inputToHidden += layerWeights[0][inputNode][hiddenNode] * activationInput[inputNode];
				}
				activationHidden[0][hiddenNode] = g(inputToHidden);
			}

			//between hiddenlayers
			for(size_t hiddenLayer = 1; hiddenLayer < numHiddenLayers; hiddenLayer++){
				activationHidden[hiddenLayer].resize(hiddenLayerSize);

				for (size_t hiddenNode = 0; hiddenNode < hiddenLayerSize; hiddenNode++)
				{
					double inputToHidden = 0;
					for (size_t inputNode = 0; inputNode < hiddenLayerSize; inputNode++)
					{
						inputToHidden += layerWeights[hiddenLayer][inputNode][hiddenNode] * activationHidden[hiddenLayer-1][inputNode];
					}
					activationHidden[hiddenLayer][hiddenNode] = g(inputToHidden);
				}
				
			}

			// output node.
			vector <double> output(outputSize);
			for(size_t outputNode = 0; outputNode < outputSize; outputNode++){
				double inputAtOutput = 0;
				for (size_t hiddenNode = 0; hiddenNode < hiddenLayerSize; hiddenNode++)
				{
					inputAtOutput += layerWeights[numHiddenLayers][hiddenNode][outputNode] * activationHidden[numHiddenLayers-1][hiddenNode];
				}
				output[outputNode] = g(inputAtOutput);
			}


			for(size_t i: output)
				cout << " " << std::setprecision(2) << output[i];


			// // calculating errors
			// double errorOfOutputNode = gprime(activationOutput) * (y[example] - activationOutput);

			// // Calculating error of hidden layer. Special calculation since we only have one output node; i.e. no summation over next layer nodes
			// // Also adjusting weights of output layer
			// vector< double > errorOfHiddenNode(hiddenLayerSize);
			// for (size_t hiddenNode = 0; hiddenNode < hiddenLayerSize; hiddenNode++)
			// {
			// 	errorOfHiddenNode[hiddenNode] = hiddenLayerWeights[hiddenNode] * errorOfOutputNode;
			// 	errorOfHiddenNode[hiddenNode] *= gprime(activationHidden[hiddenNode]);
			// }

			// //adjusting weights
			// //adjusting weights at output layer
			// for (size_t hiddenNode = 0; hiddenNode < hiddenLayerSize; hiddenNode++)
			// {
			// 	hiddenLayerWeights[hiddenNode] += alpha * activationHidden[hiddenNode] * errorOfOutputNode;
			// }

			// // Adjusting weights at hidden layer.
			// for (size_t inputNode = 0; inputNode < inputLayerSize; inputNode++)
			// {
			// 	for (size_t hiddenNode = 0; hiddenNode < hiddenLayerSize; hiddenNode++)
			// 	{
			// 		inputHiddenLayerWeights[inputNode][hiddenNode] += alpha * activationInput[inputNode] * errorOfHiddenNode[hiddenNode];
			// 	}
			// }
		}
		cout << endl;
	}

	return;
}
