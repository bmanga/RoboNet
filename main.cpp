#include "clbp/Neuron.h"
#include "clbp/Layer.h"
#include "clbp/Net.h"

#include "opencv2/opencv.hpp"

#include <iostream>

using namespace cv;
using namespace std;
constexpr int ESC_key = 27;

int main(int, char**)
{
    VideoCapture cap(0); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;

    Mat edges;
    namedWindow("edges",1);
    for(;;)
    {
        Mat frame;
        cap >> frame; // get a new frame from camera
        cvtColor(frame, edges, COLOR_BGR2GRAY);
        //GaussianBlur(edges, edges, Size(7,7), 1.5, 1.5);
        //Canny(edges, edges, 0, 30, 3);
        imshow("edges", edges);
        if(waitKey(1000) == ESC_key) break;
        cout << frame.at<float>(20, 20) << endl;
        auto intensity = frame.at<Vec3b>(100, 100);
        cout << (int)intensity[0] << " " << (int)intensity[1] << " "  << (int)intensity[2] << endl;
        //get differenceGreyValues as an array


        //pass the pointer to the Net
        //DoForward propagation
        //getPredictiveAction
        //pass the action to Robot
        //get new image
            //extract the leadError from it
            //DoBackprop of the leadError
            //extract new predictive differenceGreyValues

    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}