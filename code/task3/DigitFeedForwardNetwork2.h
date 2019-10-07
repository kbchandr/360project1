#pragma once
#include <math.h>
#include <vector>

using namespace std;

class DigitFeedForwardNetwork2
{
public:
	void initialize(int seed);

	void train(const vector< vector< double > >& x,
		const vector< double >& y, size_t numEpochs);

	void adjustWeights(const vector< vector< double > >& activationInput, const vector<double>& expectedOutput);


	DigitFeedForwardNetwork2(double alpha, size_t hiddenLayerSize, size_t numHiddenLayers, size_t inputLayerSize, size_t outputLayerSize):
		alpha(alpha), hiddenLayerSize(hiddenLayerSize), numHiddenLayers(numHiddenLayers), inputLayerSize(inputLayerSize), outputLayerSize(outputLayerSize) {}

private:
	vector<vector< vector< double > > > layerWeights; // [layer][from][to]
	vector<double> trainingLoss;
	vector<double> validationLoss;

	double alpha;
	int epochMinVLoss;
	int minVLoss;
	size_t hiddenLayerSize;
	size_t numHiddenLayers;
	size_t inputLayerSize;
	size_t outputLayerSize;

	inline double g(double x) {return 1.0 / (1.0 + exp(-x)); }
	inline double gprime(double y) {return y * (1 - y); }
};
