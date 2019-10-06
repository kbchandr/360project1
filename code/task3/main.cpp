#include <vector>
#include <iostream>

#include "MNIST_reader.h"
#include "DigitFeedForwardNetwork.h"

using namespace std;
int main()
{
	double alpha = 0.1;   // learning rate --> bigger steps
	size_t inputLayerSize = 784;
	size_t numHiddenLayers = 3; 
	size_t hiddenLayerSize = 32; // larger layersize means differentiating more
	size_t numEpochs = 7000; //time and accuracy
	size_t outputSize = 10;

	int seed = 0; // random seed for the network initialization

	string filename = "train-images.idx3-ubyte";
	//load MNIST images
	vector <vector< int > > training_images;
	vector <vector< double > > scaled_training;
	loadMnistImages(filename, training_images);
	cout << "Number of images: " << training_images.size() << endl;
	cout << "Image size: " << training_images[0].size() << endl;

	//scale
	scaleIntensity(training_images, scaled_training);

	cout << "Number of images: " << scaled_training.size() << endl;
	cout << "Image size: " << scaled_training[0].size() << endl;
	filename = "train-labels.idx1-ubyte";
	//load MNIST labels

	vector<int> training_labels;
	loadMnistLabels(filename, training_labels);
	vector<double> double_labels(training_labels.begin(), training_labels.end());

	cout << "Number of labels: " << double_labels.size() << endl;

	DigitFeedForwardNetwork nn(alpha, hiddenLayerSize, numHiddenLayers, inputLayerSize, outputSize);
	nn.initialize(seed);
	nn.train(scaled_training, double_labels, numEpochs);

	return 0;
}