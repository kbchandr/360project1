#include "DigitFeedForwardNetwork2.h"
#include <iostream>
#include <random>
#include <iomanip>      // std::setprecision

void DigitFeedForwardNetwork2::initialize(int seed)
{
	srand(seed);

	layerWeights.resize(numHiddenLayers+1); 
	//input to first hidden layer
	layerWeights[0].resize(inputLayerSize);
	for (size_t i = 0; i < inputLayerSize; i++)
	{
		layerWeights[0][i].resize(hiddenLayerSize);
		for (size_t j = 0; j < hiddenLayerSize; j++)
		{
			layerWeights[0][i][j] = (rand() % 101 - 50) * 1.0 / 1000; 	// This network cannot learn if the initial weights are set to zero.
		}
	}


	// //weights between hiddenlayers
	for(size_t i = 1; i < numHiddenLayers; i++){
		layerWeights[i].resize(hiddenLayerSize);
		for(size_t j = 0; j < hiddenLayerSize; j++){
			layerWeights[i][j].resize(hiddenLayerSize);
			for (size_t k = 0; k < hiddenLayerSize; k++)
			{
				layerWeights[i][j][k] = (rand() % 101 - 50) * 1.0 / 1000; 	// This network cannot learn if the initial weights are set to zero.
			}
		}
	}

	//weight from last hl to output layer
	layerWeights[numHiddenLayers].resize(hiddenLayerSize);
	for (size_t i = 0; i < hiddenLayerSize; i++)
	{
		layerWeights[numHiddenLayers][i].resize(outputLayerSize);
		for (size_t j = 0; j < outputLayerSize; j++)
		{
			layerWeights[numHiddenLayers][i][j] = (rand() % 101 - 50) * 1.0 / 1000; 	// This network cannot learn if the initial weights are set to zero.
		}
	}
}

void DigitFeedForwardNetwork2::train(const vector< vector< double > >& x,
	const vector< double >& y, size_t numEpochs)
{
	size_t trainingexamples = x.size();


	// train the network
	for (size_t epoch = 0; epoch < numEpochs; epoch++)
	{
		// print
		cout << "epoch = " << epoch << ", outputs = "<<endl;
		for (size_t example = 0; example < trainingexamples; example++)
		{
			// propagate the inputs forward to compute the outputs 
			vector< vector< double > > activationInput(); // We store the activation of each node (over all input and hidden layers) as we need that data during back propagation.			
			activationInput.resize(numHiddenLayers+2); //hidden layers + input + output
			activationInput[0].resize(inputLayerSize);

			for (size_t inputNode = 0; inputNode < inputLayerSize; inputNode++) // initialize input layer with training data
			{
				activationInput[0][inputNode] = x[example][inputNode];
			}
		
			for(size_t layer = 1; layer < numHiddenLayers+2; layer++){
				size_t layerSize = layerWeights[layer].size();
				double inputToHidden = 0;
				activationInput[layer].resize(layerSize);

				for(size_t to = 0; to < layerSize; to++){
					for(size_t from = 0; from < layerWeights[layer-1].size(); from++){
						inputToHidden += layerWeights[layer-1][from][to] * activationInput[layer-1][from];
					}
					activationInput[layer][to] = g(inputToHidden);
				}
			}
			
			
			cout << "output: ";
			for(size_t i = 0; i < outputLayerSize; i++)
				cout << std::fixed <<std::setprecision(3) << activationInput[numHiddenLayers+1][i]<< " ";

			cout << endl;


			// calculating errors
			vector< vector <double > > error();
			vector<double> expectedOutput(outputLayerSize);
			fill(expectedOutput.begin(), expectedOutput.end(), 0);// fill all with 0
			expectedOutput[(int)y[example]] = 1;

			cout << "expected: ";
			for(size_t i = 0; i < expectedOutput.size(); i++)
				cout << std::setprecision(2) << expectedOutput[i]<< " ";

			cout << endl;

			//calculate output error
			error.resize(numHiddenLayers+2);
			error[numHiddenLayers+1].resize(activationInput[numHiddenLayers+1].size());
			for(size_t node = 0; node < outputSize; node++){
				double actualInput = activationInput[numHiddenLayers+1][node];
				error[numHiddenLayers+1][node] = gprime(actualInput) * (expectedOutput[node] - actualInput);
			}


			for(size_t layer = numHiddenLayers; layer > 0; layer--){
				error[layer].resize(activationInput[layer].size());
				for(size_t from = 0; from < error[layer].size(); from++){
					for(size_t to = 0; to < error[layer+1].size(); to++){
						error[layer] += layerWeights[layer][from][to] * error[layer+1][to];
					}
					error[layer] *= gprime(activationInput[layer][from]);
				}
			}



			for(size_t hiddenLayer = numHiddenLayers-2; hiddenLayer > 0; hiddenLayer--){
				errorOfHiddenNodes[hiddenLayer].resize(hiddenLayerSize);

				for(size_t i = 0; i < hiddenLayerSize; i++){
					for(size_t j = 0; j < hiddenLayerSize; j++){
						errorOfHiddenNodes[hiddenLayer][i] += layerWeights[hiddenLayer][i][j] * errorOfHiddenNodes[hiddenLayer+1][j];
					}
					errorOfHiddenNodes[hiddenLayer][i] *= gprime(activationHidden[hiddenLayer][i]);

				}
			}


			//adjusting weights
			//adjusting weights at output layer

			for(size_t layer = 0; layer < numHiddenLayers+1; layer++){
				for(size_t from = 0; from < layerWeights[layer].size(); from++){
					for(size_t to = 0; to < layerWeights[layer+1].size(); to++){
						layerWeights[layer][from][to] += alpha * activationInput[layer][from] * error[layer+1][to];
					}
				}
			}

	
		}

		cout << endl;
	}

	return;
}
