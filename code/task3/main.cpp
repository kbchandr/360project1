#include <vector>
#include <iostream>

#include "MNIST_reader.h"
#include "DigitFeedForwardNetwork.h"

using namespace std;
int main()
{
	double alpha = 0.7;   // learning rate --> bigger steps
	size_t inputLayerSize = 784;
	size_t numHiddenLayers = 3; 
	size_t hiddenLayerSize = 32; // larger layersize means differentiating more
	size_t numEpochs = 7000; //time and accuracy

	int seed = 0; // random seed for the network initialization

	string filename = "train-images.idx3-ubyte";
	//load MNIST images
	vector <vector< int> > training_images;
	vector <vector<double> > scaled_training;
	loadMnistImages(filename, training_images);
	// cout << "Number of images: " << training_images.size() << endl;
	// cout << "Image size: " << training_images[0].size() << endl;

	//scale
	scaleIntensity(training_images, scaled_training);

	filename = "train-labels.idx1-ubyte";
	//load MNIST labels
	vector<int> training_labels;
	// loadMnistLabels(filename, training_labels);
	// cout << "Number of labels: " << training_labels.size() << endl;

	DigitFeedForwardNetwork nn(alpha, hiddenLayerSize, numHiddenLayers, inputLayerSize, outputSize);
	nn.initialize(seed);
	nn.train(x, y, numEpochs);

	return 0;
}