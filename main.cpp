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
static constexpr int nInputs = 2;

static constexpr int nPredictorCols = 3;
static constexpr int nPredictorRows = 4;
static constexpr int nPredictors = nPredictorCols * nPredictorRows;

static constexpr double constantSpeed = 10;

int nNeurons[nLayers] = { nPredictors, 1 };

Net net{ nLayers, nNeurons, nInputs };



int16_t onStepCompleted(int deltaSensorData, const std::vector<double> predictorDeltas)
{
	double errorGain = 1;
	double error = errorGain * deltaSensorData;



	int gain = 50;

	cout << "MAIN PROGRAM: NEXT ITERATION" << endl;
	net.setInputs(predictorDeltas.data());
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


    std::vector<double> predictorDeltaMeans;
    predictorDeltaMeans.reserve(nPredictorCols * nPredictorRows);
	
	for (;;)
	{
	  predictorDeltaMeans.clear();

		Mat frame;
		cap >> frame; // get a new frame from camera
		cvtColor(frame, edges, COLOR_BGR2GRAY);

		// Define the rect area that we want to consider.

		int areaWidth = 500;
		int startX = (frame.cols - areaWidth) / 2;
		auto area = Rect{startX, 150, areaWidth, 300};

		int predictorWidth = area.width / 2 / nPredictorCols;
		int predictorHeight = area.height / nPredictorRows;

		rectangle(edges, area, Scalar(122, 144, 255));

    int areaMiddleLine = area.width / 2 + area.x;


    for (int k = 0; k < nPredictorRows; ++k) {
		  for (int j = 0; j < nPredictorCols; ++j) {
        auto lPred = Rect(areaMiddleLine - (j + 1) * predictorWidth, area.y + k * predictorHeight, predictorWidth, predictorHeight);
        auto rPred = Rect(areaMiddleLine + (j) * predictorWidth, area.y + k * predictorHeight, predictorWidth, predictorHeight);

        predictorDeltaMeans.push_back(mean(Mat(edges, lPred))[0] - mean(Mat(edges,rPred))[0]);

        rectangle(edges, lPred, Scalar(100, 100, 100));
        rectangle(edges, rPred, Scalar(100, 100, 100));
		  }
		}

    cvtColor(edges, frame, CV_GRAY2RGB);

    line(frame, {areaMiddleLine, area.tl().y}, {areaMiddleLine, area.br().y}, Scalar(50, 50, 255));


		imshow("visual", frame);
		
		int8_t deltaSensor = 0;
		Ret = LS.Read(&deltaSensor, sizeof(deltaSensor));

		if (Ret > 0) {
			cout << "delta sensor is " << (int)deltaSensor << std::endl;
			int16_t error = onStepCompleted(deltaSensor, predictorDeltaMeans);
			Ret = LS.Write(&error, sizeof(error));
		}

		if (waitKey(20) == ESC_key) break;
	}
    return 0;
}