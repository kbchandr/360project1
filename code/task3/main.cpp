#include <vector>
#include <iostream>

#include "MNIST_reader.h"

using namespace std;
int main()
{
	string filename = "../MNIST/train-images.idx3-ubyte";
	//load MNIST images
	vector <vector< int> > training_images;
	loadMnistImages(filename, training_images);
	cout << "Number of images: " << training_images.size() << endl;
	cout << "Image size: " << training_images[0].size() << endl;

	filename = "../MNIST/train-labels.idx1-ubyte";
	//load MNIST labels
	vector<int> training_labels;
	loadMnistLabels(filename, training_labels);
	cout << "Number of labels: " << training_labels.size() << endl;

	return 0;
}