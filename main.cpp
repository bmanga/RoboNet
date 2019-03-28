#include "clbp/Neuron.h"
#include "clbp/Layer.h"
#include "clbp/Net.h"

#include "opencv2/opencv.hpp"
#include "serialib.h"

#include <iostream>

using namespace cv;
using namespace std;
constexpr int ESC_key = 27;

static constexpr int nLayers = 2;
static constexpr int nInputs = 4;
static constexpr int nPredictors = 4;

static constexpr double constantSpeed = 30;
int nNeurons[nLayers] = { 2, 1 };
Net net{ nLayers, nNeurons, nInputs };



int space = 96;
int x_1 = space;
int y_1 = 120;
int xboxsize = 40;
int yboxsize = y_1 + xboxsize;
int x_2 = x_1 + xboxsize + space;
int x_3 = 640-space-xboxsize; // 4th square
int x_4 = x_3 - space - xboxsize;


int16_t onStepCompleted(int deltaSensorData, double *predictors)
{
	double errorGain = 1;
	double error = errorGain * deltaSensorData;



	int gain = 100;

	cout << "MAIN PROGRAM: NEXT ITERATION" << endl;
	net.setInputs(predictors);
	double learningRate = 0.01;
	net.setLearningRate(learningRate);
	net.propInputs();
	double leadError = error;
	net.setError(leadError);
	net.propError();
	net.updateWeights();
	//need to do weight change first
	net.saveWeights();
	double icoOutput = net.getOutput(0);
	double error2 = (error + icoOutput) * gain;
	std::cout << "error out is: " << error2 << std::endl;
	return (int16_t)(error2 * 1);
}

#if defined (_WIN32) || defined( _WIN64)
#define         DEVICE_PORT             "COM4"                               // COM1 for windows
#endif

#ifdef __linux__
#define         DEVICE_PORT             "/dev/ttyS0"                         // ttyS0 for linux

#endif


int main(int, char**)
{
	serialib LS;
	char Ret = LS.Open(DEVICE_PORT, 9600);
	if (Ret != 1) {                                                           // If an error occured...
		printf("Error while opening port. Permission problem ?\n");        // ... display a message ...
		return Ret;                                                         // ... quit the application
	}
	printf("Serial port opened successfully !\n");
    VideoCapture cap(1); // open the default camera
	cap.set(CAP_PROP_FPS, 30);
    if(!cap.isOpened())  // check if we succeeded
        return -1;

    Mat edges;
    namedWindow("edges",1);
	
	for (;;)
	{
		//cap.set(CAP_PROP_GAIN, yeet);		
		//1) get new image
		Mat frame;
		cap >> frame; // get a new frame from camera
		cvtColor(frame, edges, COLOR_BGR2GRAY);
		//GaussianBlur(edges, edges, Size(7,7), 1.5, 1.5);
		//Canny(edges, edges, 0, 30, 3);

		//Draw rectangles 
		rectangle(edges, Point(x_1, y_1), Point(x_1 + xboxsize, yboxsize), Scalar(100, 0, 255), 2, 8);
		rectangle(edges, Point(x_2, y_1), Point(x_2 + xboxsize, yboxsize), Scalar(100, 0, 255), 2, 8);
		rectangle(edges, Point(x_3, y_1), Point(x_3 + xboxsize, yboxsize), Scalar(100, 0, 255), 2, 8);
		rectangle(edges, Point(x_4, y_1), Point(x_4 + xboxsize, yboxsize), Scalar(100, 0, 255), 2, 8);

		Mat square1(edges, Rect(x_1, y_1, xboxsize, xboxsize));
		Mat square2(edges, Rect(x_2, y_1, xboxsize, xboxsize));
		Mat square3(edges, Rect(x_4, y_1, xboxsize, xboxsize));
		Mat square4(edges, Rect(x_3, y_1, xboxsize, xboxsize));

		Scalar value1 = mean(square1);
		Scalar value2 = mean(square2);
		Scalar value3 = mean(square3);
		Scalar value4 = mean(square4);

		double predictors[nPredictors] = { value1[0], value2[0], value3[0], value4[0] };
		//cout << "value 1: " << value1[0] << "       value 2:  " << value2[0] << "       value 3:  " << value3[0] << "       value 4:  " << value4[0] << endl;
		imshow("edges", edges);
		
		int8_t deltaSensor = 0;
		Ret = LS.Read(&deltaSensor, sizeof(deltaSensor));

		if (Ret > 0) {
			cout << "delta sensor is " << (int)deltaSensor << std::endl;
			int16_t error = onStepCompleted(deltaSensor, predictors);
			Ret = LS.Write(&error, sizeof(error));
		}

		if (waitKey(20) == ESC_key) break;
		//cout << frame.at<float>(20, 20) << endl;
		//auto intensity = frame.at<Vec3b>(100, 100);
		//cout << (int)intensity[0] << " " << (int)intensity[1] << " "  << (int)intensity[2] << endl;
		//extract the leadError from it or read the error from the dedicated error sensors
		//DoBackprop of the leadError
		//get differenceGreyValues as an array
		//pass the above array to the Net as a pointer
		//DoForward propagation
		//getPredictiveAction
		//pass the action to Robot
	}
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}