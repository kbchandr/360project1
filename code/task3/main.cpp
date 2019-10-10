#include <vector>
#include <iostream>
#include <time.h>

#include "MNIST_reader.h"
#include "DigitFeedForwardNetwork2.h"
#include "BinaryNetwork.h"

using namespace std;
int main()
{
	double alpha = 0.1;   // learning rate --> bigger steps
	size_t inputLayerSize = 784;
	size_t numHiddenLayers = 2; 
	size_t hiddenLayerSize = 32; // larger layersize means differentiating more
	size_t numEpochs = 25; //time and accuracy
	size_t outputSize = 10;

	int seed = 21; // random seed for the network initialization
	cout <<alpha<< " "<<numHiddenLayers<<" "<<hiddenLayerSize<<" "<<numEpochs<<endl;
	string filename = "train-images.idx3-ubyte";
	//load MNIST images
	vector <vector< int > > training_images;
	vector <vector< double > > scaled_training;
	loadMnistImages(filename, training_images);
	training_images.resize(6000);
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
	training_labels.resize(6000);
	vector<double> double_labels(training_labels.begin(), training_labels.end());

	cout << "Number of labels: " << double_labels.size() << endl;

	// BinaryNetwork nn(alpha, hiddenLayerSize, numHiddenLayers, inputLayerSize, outputSize);
	DigitFeedForwardNetwork2 nn(alpha, hiddenLayerSize, numHiddenLayers, inputLayerSize, outputSize);
	nn.initialize(seed);
		clock_t t=clock();

	nn.train(scaled_training, double_labels, numEpochs);
	clock_t s = clock();
	cout <<(float)(s-t)/CLOCKS_PER_SEC;

	filename = "t10k-images.idx3-ubyte";
	vector <vector< int > > testing_images;
	vector <vector< double > > scaled_testing;
	loadMnistImages(filename, testing_images);
	scaleIntensity(testing_images, scaled_testing);
	filename = "t10k-labels.idx1-ubyte";
	vector<int> testing_labels;
	loadMnistLabels(filename, testing_labels);
	vector<double> double_test(testing_labels.begin(), testing_labels.end());
	nn.test(scaled_testing, double_test);



	return 0;
}