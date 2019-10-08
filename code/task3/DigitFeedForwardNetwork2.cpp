#include "DigitFeedForwardNetwork2.h"
#include <climits>
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

void DigitFeedForwardNetwork2::adjustWeights(const vector< vector< double > >& activationInput, const vector<double>& expectedOutput){
	//calculate output error
	vector< vector <double > > error;

	error.resize(numHiddenLayers+2);
	error[numHiddenLayers+1].resize(activationInput[numHiddenLayers+1].size());
	for(size_t node = 0; node < outputLayerSize; node++){
		double actualInput = activationInput[numHiddenLayers+1][node];
		error[numHiddenLayers+1][node] = gprime(actualInput) * (expectedOutput[node] - actualInput);
	}


	for(size_t layer = numHiddenLayers; layer > 0; layer--){
		error[layer].resize(activationInput[layer].size());
		for(size_t from = 0; from < error[layer].size(); from++){
			for(size_t to = 0; to < error[layer+1].size(); to++){
				error[layer][from] += layerWeights[layer][from][to] * error[layer+1][to];
			}
			error[layer][from] *= gprime(activationInput[layer][from]);
		}
	}

	

	//adjusting weights
	//adjusting weights at output layer

	for(size_t layer = 0; layer < numHiddenLayers+1; layer++){
		for(size_t from = 0; from < layerWeights[layer].size(); from++){
			for(size_t to = 0; to < layerWeights[layer][from].size(); to++){
				layerWeights[layer][from][to] += alpha * activationInput[layer][from] * error[layer+1][to];
			}
		}
	}
}

vector< vector< double > > DigitFeedForwardNetwork2::feedForward(const vector< vector< double > >& x, size_t example){
	vector< vector< double > > activationInput; // We store the activation of each node (over all input and hidden layers) as we need that data during back propagation.			

	activationInput.resize(numHiddenLayers+2); //hidden layers + input + output
	activationInput[0].resize(inputLayerSize);

	for (size_t inputNode = 0; inputNode < inputLayerSize; inputNode++) // initialize input layer with training data
	{
		activationInput[0][inputNode] = x[example][inputNode];
	}

	for(size_t layer = 1; layer < numHiddenLayers+1; layer++){
		size_t layerSize = layerWeights[layer].size();
		double inputToHidden = 0;
		activationInput[layer].resize(layerSize);
		for(size_t to = 0; to < layerSize; to++){
			for(size_t from = 0; from < activationInput[layer-1].size(); from++){
				inputToHidden += layerWeights[layer-1][from][to] * activationInput[layer-1][from];
			}
			activationInput[layer][to] = g(inputToHidden);
		}

	}

	//output
	activationInput[numHiddenLayers+1].resize(outputLayerSize);
	for(size_t to = 0; to < outputLayerSize; to++){
		double inputToHidden = 0;
		for(size_t from = 0; from < hiddenLayerSize; from++){
			inputToHidden += layerWeights[numHiddenLayers][from][to] * activationInput[numHiddenLayers][from];
		}
		activationInput[numHiddenLayers+1][to] = g(inputToHidden);
	}
			
	return activationInput;
}

void DigitFeedForwardNetwork2::train(const vector< vector< double > >& x,
	const vector< double >& y, size_t numEpochs)
{


	// train the network
	for (size_t epoch = 0; epoch < numEpochs; epoch++)
	{
		double totalTrainSamples = 0;
		double correctTrainSamples = 0;
		double totalValSamples = 0;
		double correctValSamples = 0;
		double trainLoss= 0;
		double valLoss = 0;
		// print
		cout << "epoch = " << epoch <<endl;
		for (size_t example = 0; example < 4000; example++)
		{
			// propagate the inputs forward to compute the outputs 
			vector< vector< double > > activationInput;
			activationInput = feedForward(x, example);

			// calculating errors
			vector<double> expectedOutput(outputLayerSize);
			fill(expectedOutput.begin(), expectedOutput.end(), 0);// fill all with 0
			expectedOutput[(int)y[example]] = 1;

			int predictedOutputIndex = 0;
			for(int i = 1; i < 10; i++){
				if(activationInput[numHiddenLayers+1][i] > activationInput[numHiddenLayers+1][predictedOutputIndex]){
					predictedOutputIndex = i;
				}
			}
			//calculate loss
			double loss = 0;
			for(size_t i = 0; i < expectedOutput.size(); i++){
				double y = expectedOutput[i];
				double o = activationInput[numHiddenLayers+1][i];
				loss += (y-o) * (y-o);
			}
			
			
			adjustWeights(activationInput, expectedOutput);
			trainLoss+=loss;
			totalTrainSamples++;
			if(predictedOutputIndex == y [example]){
				correctTrainSamples++;
			}
			
			


	
		}

		// for(size_t example = 4000; example < 6000; example++){
		// 	vector< vector< double > > activationInput;
		// 	activationInput = feedForward(x, example);

		// 	// calculating errors
		// 	vector<double> expectedOutput(outputLayerSize);
		// 	fill(expectedOutput.begin(), expectedOutput.end(), 0);// fill all with 0
		// 	expectedOutput[(int)y[example]] = 1;

		// 	int predictedOutputIndex = 0;
		// 	for(int i = 1; i < 10; i++){
		// 		if(activationInput[numHiddenLayers+1][i] > activationInput[numHiddenLayers+1][predictedOutputIndex]){
		// 			predictedOutputIndex = i;
		// 		}
		// 	}
		// 	//calculate loss
		// 	double loss = 0;
		// 	for(size_t i = 0; i < expectedOutput.size(); i++){
		// 		double y = expectedOutput[i];
		// 		double o = activationInput[numHiddenLayers+1][i];
		// 		loss += (y-o) * (y-o);
		// 	}
		// 	valLoss +=loss;
		// 	totalValSamples++;				
		// 	if(predictedOutputIndex == y [example]){
		// 		correctValSamples++;
		// 	}
		// }
		// cout << endl;
		// cout << "Training Accuracy: "<< correctTrainSamples/totalTrainSamples<<endl;
		// cout << "Validation Accuracy: "<<correctValSamples/totalValSamples <<endl;
		// cout << "Training Loss: "<< trainLoss/totalTrainSamples<<endl;
		// cout << "Validation Loss: "<<valLoss/totalValSamples <<endl;

	}


	return;
}


void DigitFeedForwardNetwork2::test(const vector< vector< double > >& x,
	const vector< double >& y)
{


	// train the network

		double totalTestSamples = 0;
		double correctTestSamples = 0;


		// print
		for (size_t example = 0; example < 10000; example++)
		{
			// propagate the inputs forward to compute the outputs 
			vector< vector< double > > activationInput;
			activationInput = feedForward(x, example);

			// calculating errors
			vector<double> expectedOutput(outputLayerSize);
			fill(expectedOutput.begin(), expectedOutput.end(), 0);// fill all with 0
			expectedOutput[(int)y[example]] = 1;

			int predictedOutputIndex = 0;
			for(int i = 1; i < 10; i++){
				if(activationInput[numHiddenLayers+1][i] > activationInput[numHiddenLayers+1][predictedOutputIndex]){
					predictedOutputIndex = i;
				}
			}

			totalTestSamples++;
			if(predictedOutputIndex == y [example]){
				correctTestSamples++;
			}
			
			


	
		}

		cout << endl;
		cout << "Test Accuracy: "<< correctTestSamples/totalTestSamples<<endl;


	


	return;
}