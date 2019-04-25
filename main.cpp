//#include "clbp/Neuron.h"
//#include "clbp/Layer.h"
//#include "clbp/Net.h"
#include "opencv2/opencv.hpp"
#include "serialib.h"

#include <iostream>
#include <string>
#include <chrono>
#include <boost/circular_buffer.hpp>

#include "neural.h"

using namespace cv;
using namespace std;
constexpr int ESC_key = 27;

static constexpr int nLayers = 3;

static constexpr int nPredictorCols = 6;
static constexpr int nPredictorRows = 1;
static constexpr int nPredictors = nPredictorCols * nPredictorRows;

static constexpr double constantSpeed = 10;

int nNeurons[nLayers] = { nPredictors, 5, 1 };

//Net net{ nLayers, nNeurons, nPredictors };

//


int16_t onStepCompleted(double deltaSensorData, std::vector<float> &predictorDeltas)
{
  double errorGain = 5;
  double error = errorGain * deltaSensorData;

  int gain = 60;

  //cout << "MAIN PROGRAM: NEXT ITERATION" << endl;
  //net.setInputs(predictorDeltas.data());
  //double learningRate = 0.01;
  //net.setLearningRate(learningRate);
  //net.propInputs();


  //net.setError(-leadError);
  //net.propError();
  //net.updateWeights();
  //need to do weight change first
  //net.saveWeights();

  double result = run_samanet(predictorDeltas, deltaSensorData);
  double error2 = (error / 4 + result * 2.5 ) * gain;
  return (int16_t)(error2 * 0.5);

}

#if defined (_WIN32) || defined( _WIN64)
#define         DEVICE_PORT             "COM4"                               // COM1 for windows
#endif

#ifdef __linux__
#define         DEVICE_PORT             "/dev/ttyUSB0"                         // ttyS0 for linux

#endif


double calculateErrorValue(Mat &frame, Mat &output)
{
  constexpr int numErrorSensors = 5;
  int areaWidth = 400;
  int areaHeight = 30;
  int offsetFromBottom = 0;
  int whiteSensorThreshold = 210;
  int startX = (frame.cols - areaWidth) / 2;
  auto area = Rect{ startX, frame.rows - areaHeight - offsetFromBottom, areaWidth, areaHeight };

  int areaMiddleLine = area.width / 2 + area.x;

  int sensorWidth = area.width / 2 / numErrorSensors;
  int sensorHeight = areaHeight;

  std::array<double, numErrorSensors> sensorWeights;

  // Linear weights, maybe consider exponential.
  double increase = 1.0 / numErrorSensors;
  sensorWeights[numErrorSensors - 1] = 1;
  for (int j = numErrorSensors - 2; j > 0; --j) {
    sensorWeights[j] = sensorWeights[j + 1] * 0.5;
  }



  int numTriggeredPairs = 0;
  double error = 0;


  for (int j = 0; j < numErrorSensors; ++j) {
    auto lPred = Rect(areaMiddleLine - (j + 1) * sensorWidth, area.y, sensorWidth, sensorHeight);
    auto rPred = Rect(areaMiddleLine + (j)* sensorWidth, area.y, sensorWidth, sensorHeight);

    double grayMeanL = (mean(Mat(frame, lPred))[0]) > whiteSensorThreshold;
    double grayMeanR = (mean(Mat(frame, rPred))[0]) > whiteSensorThreshold;

    auto diff = (grayMeanR - grayMeanL);
    numTriggeredPairs += (diff != 0);

    error += diff * sensorWeights[j];


    //predictorDeltaMeans.push_back((grayMeanL - grayMeanR) / 255);
    putText(output, std::to_string((int)grayMeanL), Point{ lPred.x + lPred.width / 2, lPred.y + lPred.height / 2 }, FONT_HERSHEY_SIMPLEX, 0.4, { 255, 100, 100 });
    putText(output, std::to_string((int)grayMeanR), Point{ rPred.x + rPred.width / 2, rPred.y + rPred.height / 2 }, FONT_HERSHEY_SIMPLEX, 0.4, { 255, 100, 100 });
    rectangle(output, lPred, Scalar(100, 100, 100));
    rectangle(output, rPred, Scalar(100, 100, 100));
  }

  return numTriggeredPairs ? error / numTriggeredPairs : 0;
}

int main(int, char**)
{
  initialize_samanet();
//  net.initWeights(Neuron::W_ONES, Neuron::B_NONE);
  serialib LS;
  char Ret = LS.Open(DEVICE_PORT, 115200);
  if (Ret != 1) {                                                           // If an error occured...
    printf("Error while opening port. Permission problem ?\n");        // ... display a message ...
    return Ret;                                                         // ... quit the application
  }
  printf("Serial port opened successfully !\n");
  VideoCapture cap(0); // open the default camera
  //cap.set(CAP_PROP_FPS, 30);
  cap.set(CAP_PROP_FOURCC, CV_FOURCC('M', 'J', 'P', 'G'));
  if (!cap.isOpened())  // check if we succeeded
    return -1;

  Mat edges;
  namedWindow("edges", 1);



  std::vector<float> predictorDeltaMeans;
  predictorDeltaMeans.reserve(nPredictorCols * nPredictorRows);

  for (;;)
  {
    predictorDeltaMeans.clear();

    Mat frame;
    cap >> frame; // get a new frame from camera
    cvtColor(frame, edges, COLOR_BGR2GRAY);


    //std::cout << "calculated error from image is: " << err << "\n";


    // Define the rect area that we want to consider.

    int areaWidth = 500;
    int areaHeight = 30;
    int offsetFromTop = 200;
    int startX = (frame.cols - areaWidth) / 2;
    auto area = Rect{ startX, offsetFromTop, areaWidth, areaHeight };

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
        putText(frame, std::to_string((int)grayMeanL), Point{ lPred.x + lPred.width / 2, lPred.y + lPred.height / 2 }, FONT_HERSHEY_SIMPLEX, 0.4, { 255, 255, 255 });
        putText(frame, std::to_string((int)grayMeanR), Point{ rPred.x + rPred.width / 2, rPred.y + rPred.height / 2 }, FONT_HERSHEY_SIMPLEX, 0.4, { 255, 255, 255 });
        rectangle(frame, lPred, Scalar(100, 100, 100));
        rectangle(frame, rPred, Scalar(100, 100, 100));
      }
    }

    //cvtColor(edges, frame, COLOR_GRAY2RGB);

    line(frame, { areaMiddleLine, area.tl().y }, { areaMiddleLine, area.br().y }, Scalar(50, 50, 255));

    double err = calculateErrorValue(edges, frame);
    imshow("robot view", frame);

    int8_t ping = 0;
    Ret = LS.Read(&ping, sizeof(ping));


    if (Ret > 0) {
      //cout << "delta sensor is " << (int)deltaSensor << std::endl;
      //cout << "image sensor is " << err << std::endl;
      //if (deltaSensor != 0) system("mpv /usr/share/sounds/freedesktop/stereo/bell.oga");

      int16_t error = onStepCompleted(err, predictorDeltaMeans);
      //int16_t error = deltaSensor * 50;

      Ret = LS.Write(&error, sizeof(error));
    }

    if (waitKey(20) == ESC_key) break;
  }
  return 0;
}