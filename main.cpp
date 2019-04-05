#include "clbp/Neuron.h"
#include "clbp/Layer.h"
#include "clbp/Net.h"
#include "opencv2/opencv.hpp"
#include "serialib.h"

#include <iostream>

using namespace cv;
using namespace std;
constexpr int ESC_key = 27;

static constexpr int nLayers = 3;

static constexpr int nPredictorCols = 3;
static constexpr int nPredictorRows = 4;
static constexpr int nPredictors = nPredictorCols * nPredictorRows;

static constexpr double constantSpeed = 10;

int nNeurons[nLayers] = { nPredictors, 5, 1 };

Net net{ nLayers, nNeurons, nPredictors };



int16_t onStepCompleted(int deltaSensorData, const std::vector<double> &predictorDeltas)
{
  double errorGain = 5;
  double error = errorGain * deltaSensorData;



  int gain = 30;

  //cout << "MAIN PROGRAM: NEXT ITERATION" << endl;
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
  double error2 = (error / 2 + net.getOutput(0)) * gain;
  std::cout << "neural output is: " << net.getOutput(0) << std::endl;
  return (int16_t)(error2 * 0.5);
}

#if defined (_WIN32) || defined( _WIN64)
#define         DEVICE_PORT             "COM4"                               // COM1 for windows
#endif

#ifdef __linux__
#define         DEVICE_PORT             "/dev/ttyS0"                         // ttyS0 for linux

#endif


int main(int, char**)
{
  net.initWeights(Neuron::W_ONES, Neuron::B_NONE);
  serialib LS;
  char Ret = LS.Open(DEVICE_PORT, 115200);
  if (Ret != 1) {                                                           // If an error occured...
    printf("Error while opening port. Permission problem ?\n");        // ... display a message ...
    return Ret;                                                         // ... quit the application
  }
  printf("Serial port opened successfully !\n");
  VideoCapture cap(1); // open the default camera
  cap.set(CAP_PROP_FPS, 30);
  if (!cap.isOpened())  // check if we succeeded
    return -1;

  Mat edges;
  namedWindow("edges", 1);


  std::vector<double> predictorDeltaMeans;
  predictorDeltaMeans.reserve(nPredictorCols * nPredictorRows);

  for (;;)
  {
    predictorDeltaMeans.clear();

    Mat frame;
    cap >> frame; // get a new frame from camera
    cvtColor(frame, edges, COLOR_BGR2GRAY);

    // Define the rect area that we want to consider.

    int areaWidth = 300;
    int startX = (frame.cols - areaWidth) / 2;
    auto area = Rect{ startX, 220, areaWidth, 200 };

    int predictorWidth = area.width / 2 / nPredictorCols;
    int predictorHeight = area.height / nPredictorRows;

    rectangle(edges, area, Scalar(122, 144, 255));

    int areaMiddleLine = area.width / 2 + area.x;


    for (int k = 0; k < nPredictorRows; ++k) {
      for (int j = 0; j < nPredictorCols; ++j) {
        auto lPred = Rect(areaMiddleLine - (j + 1) * predictorWidth, area.y + k * predictorHeight, predictorWidth, predictorHeight);
        auto rPred = Rect(areaMiddleLine + (j)* predictorWidth, area.y + k * predictorHeight, predictorWidth, predictorHeight);

        auto grayMeanL = mean(Mat(edges, lPred))[0];
        auto grayMeanR = mean(Mat(edges, rPred))[0];
        predictorDeltaMeans.push_back((grayMeanL - grayMeanR) / 255);
        putText(edges, std::to_string((int)grayMeanL), Point{ lPred.x + lPred.width / 2, lPred.y + lPred.height / 2 }, FONT_HERSHEY_SIMPLEX, 0.4, { 255, 255, 255 });
        putText(edges, std::to_string((int)grayMeanR), Point{ rPred.x + rPred.width / 2, rPred.y + rPred.height / 2 }, FONT_HERSHEY_SIMPLEX, 0.4, { 255, 255, 255 });
        rectangle(edges, lPred, Scalar(100, 100, 100));
        rectangle(edges, rPred, Scalar(100, 100, 100));
      }
    }

    cvtColor(edges, frame, COLOR_GRAY2RGB);

    line(frame, { areaMiddleLine, area.tl().y }, { areaMiddleLine, area.br().y }, Scalar(50, 50, 255));


    imshow("robot view", frame);

    int8_t deltaSensor = 0;
    Ret = LS.Read(&deltaSensor, sizeof(deltaSensor));

    if (Ret > 0) {
      cout << "delta sensor is " << (int)deltaSensor << std::endl;
      int16_t error = onStepCompleted(deltaSensor, predictorDeltaMeans);
      //int16_t error = deltaSensor * 50;

      Ret = LS.Write(&error, sizeof(error));
      std::cout << "error out is: " << error << std::endl;
    }

    if (waitKey(20) == ESC_key) break;
  }
  return 0;
}