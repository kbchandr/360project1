#pragma once
#include <math.h>
#include <vector>

using namespace std;

class DigitFeedForwardNetwork2
{
public:
	void initialize(int seed);
	double generateRandomWeight();
	vector< vector< double > > feedForward(const vector< vector< double > >& x, size_t example);
	void train(const vector< vector< double > >& x,
		const vector< double >& y, size_t numEpochs);
	void test(const vector< vector< double > >& x, const vector< double >& y);
	void adjustWeights(const vector< vector< double > >& activationInput, const vector<double>& expectedOutput);


	DigitFeedForwardNetwork2(double alpha, size_t hiddenLayerSize, size_t numHiddenLayers, size_t inputLayerSize, size_t outputLayerSize):
		alpha(alpha), hiddenLayerSize(hiddenLayerSize), numHiddenLayers(numHiddenLayers), inputLayerSize(inputLayerSize), outputLayerSize(outputLayerSize) {}

private:
	vector<vector< vector< double > > > layerWeights; // [layer][from][to]

	double alpha;
	size_t hiddenLayerSize;
	size_t numHiddenLayers;
	size_t inputLayerSize;
	size_t outputLayerSize;

	//sigmoid
	// inline double g(double x) {return 1.0 / (1.0 + exp(-x)); }
	// inline double gprime(double y) {return y * (1 - y); }

	//tanh
	inline double g(double x) {return (exp(x)-exp(-x))/ (exp(x)+exp(-x)); }
	inline double gprime(double y) {return 1- (g(y) *g(y)); }
};
