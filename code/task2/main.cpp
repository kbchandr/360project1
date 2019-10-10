#include <vector>
#include <iostream>
#include <time.h>
#include "SimpleFeedForwardNetwork.h"


using namespace std;
int main()
{
	clock_t t=clock();
	// hyper-paramters
	double alpha = 10;   // learning rate --> bigger steps
	size_t inputLayerSize = 2; 
	size_t hiddenLayerSize = 7; // larger layersize means differentiating more
	size_t numEpochs = 1000; //time and accuracy

	int seed = 0; // random seed for the network initialization

	// input data
	vector< vector< double > > x(4);
	x[0].push_back(0);
	x[0].push_back(0);
	x[1].push_back(0);
	x[1].push_back(1);
	x[2].push_back(1);
	x[2].push_back(0);
	x[3].push_back(1);
	x[3].push_back(1);
	vector< double > y{ 0, 1, 1, 0 };


	SimpleFeedForwardNetwork nn(alpha, hiddenLayerSize, inputLayerSize);
	nn.initialize(seed);

	nn.train(x, y, numEpochs);
	clock_t s = clock();
	cout <<(float)(s-t)/CLOCKS_PER_SEC;
	return 0;
}
