#pragma once
#include <math.h>
#include <vector>

using namespace std;

class DigitFeedForwardNetwork
{
public:
	void initialize(int seed);

	void train(const vector< vector< double > >& x,
		const vector< double >& y, size_t numEpochs);

	DigitFeedForwardNetwork(double alpha, size_t hiddenLayerSize, size_t numHiddenLayers, size_t inputLayerSize, size_t outputSize) :
		alpha(alpha), hiddenLayerSize(hiddenLayerSize), numHiddenLayers(numHiddenLayers), inputLayerSize(inputLayerSize), outputSize(outputSize) {}

private:
	vector<vector< vector< double > > > layerWeights; // [layer][from][to]

	double alpha;
	size_t hiddenLayerSize;
	size_t inputLayerSize;
	size_t numHiddenLayers;
	size_t outputSize;

	inline double g(double x) {return 1.0 / (1.0 + exp(-x)); }
	inline double gprime(double y) {return y * (1 - y); }
};