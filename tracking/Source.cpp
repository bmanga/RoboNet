#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>

#include <math.h>
#include <vector>
#include <stdlib.h>

using namespace cv;
using namespace std;

constexpr int ESC_key = 27;
int h_lower = 0;
int h_upper = 0;
int s_lower = 0;
int s_upper = 0;
int v_lower = 0;
int v_upper = 0;
int s = 100;
int v = 100;
const int alpha_slider_max = 179;
int h_lower_slider, h_upper_slider, s_lower_slider, s_upper_slider, v_lower_slider, v_upper_slider,c_colour;
double alpha;
double beta;
int colour = 0;
int pos = 0;
int green = 100000;
int blue = 1000000;
Mat frame;

static void on_trackbar_lower_h(int, void*)
{
  h_lower = h_lower_slider;
} 
static void on_trackbar_upper_h(int, void*)
{
  h_upper = h_upper_slider;
}
static void on_trackbar_lower_s(int, void*)
{
  s_lower = s_lower_slider;
}
static void on_trackbar_upper_s(int, void*)
{
  s_upper = s_upper_slider;
}
static void on_trackbar_lower_v(int, void*)
{
  v_lower = v_lower_slider;
}
static void on_trackbar_upper_v(int, void*)
{
  v_upper = v_upper_slider;
}
static void colour_change(int, void*)
{
  if (c_colour == 1) {
    green = pos;
    imwrite("C:/Users/Callum/Documents/RoboNet/tracking/traces/Red.png", frame);
  }
  if (c_colour == 2) {
    imwrite("C:/Users/Callum/Documents/RoboNet/tracking/traces/Green.png", frame);
    blue = pos;
  }
}

int main() {
  float speed = 0.0;
  int previous_diff = 0;
  ofstream myfile;
  VideoCapture cap(1); // Open the camera
  Mat frame_hsv, frame_threshold, frame1;
  int thresh = 100;
  double areas;
  Point center;
  Point previous_center;
  previous_center.x = 0;
  previous_center.y = 0;
  const int SIZE = 100000;
  Point trace[SIZE];

  VideoWriter video("recording.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, Size(640, 480));
  while (1) {
    namedWindow("Sliders", WINDOW_AUTOSIZE); // Create Window
    // Create a series of sliders
    createTrackbar("H Lower", "Sliders", &h_lower_slider, 179, on_trackbar_lower_h);
    createTrackbar("H Upper", "Sliders", &h_upper_slider, 179, on_trackbar_upper_h);
    createTrackbar("S Lower", "Sliders", &s_lower_slider, 255, on_trackbar_lower_s);
    createTrackbar("S Upper", "Sliders", &s_upper_slider, 255, on_trackbar_upper_s);
    createTrackbar("V Lower", "Sliders", &v_lower_slider, 255, on_trackbar_lower_v);
    createTrackbar("V Upper", "Sliders", &v_upper_slider, 255, on_trackbar_upper_v);
    createTrackbar("Colour", "Sliders", &c_colour, 2, colour_change);

    cap >> frame1;
    flip(frame1, frame, 1);
    float mpp = (2.24 * 100) / 640;
    cvtColor(frame, frame_hsv, COLOR_BGR2HSV);
    blur(frame_hsv, frame_hsv, Size(3, 3));
    imshow("HSV", frame_hsv);

    //Fixed values
    inRange(frame_hsv, Scalar(0, 59, 255), Scalar(27, 255, 255), frame_threshold);
    //If using sliders
    //inRange(frame_hsv, Scalar(h_lower, s_lower, v_lower), Scalar(h_upper, s_upper, v_upper), frame_threshold);

    // Find contours from binary image 
    int i;
    vector<Vec4i> hierarchy;
    vector<vector<Point> > contours;
    findContours(frame_threshold, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    vector<float>areas(contours.size());
    //find largest contour area  
    for (i = 0; i < contours.size(); i++)
    {
      areas[i] = contourArea(Mat(contours[i]));
    }

    //get index of largest contour  
    double max;
    Point maxPosition;
    minMaxLoc(Mat(areas), 0, &max, 0, &maxPosition);

    //draw largest contour
    drawContours(frame_threshold, contours, maxPosition.y, Scalar(255), cv::FILLED);
    imshow("LargestContour", frame_threshold);

    //draw bounding rectangle around largest contour  
    Rect r;
    if (contours.size() >= 1)
    {
      r = boundingRect(contours[maxPosition.y]);
      rectangle(frame, r.tl(), r.br(), CV_RGB(255, 0, 0), 3, 8, 0); //draw rectangle  


      //get centre  
      center.x = r.x + (r.width / 2);
      center.y = r.y + (r.height / 2);
      if (abs(previous_center.x - center.x) < 100 && abs(previous_center.y - center.y) < 100 || (previous_center.x == 0 && previous_center.y == 0)) {
        //Plot a circle in the centre

        circle(frame, center, 1, Scalar(0, 0, 255), 10, 9);
        float travel_x = abs(previous_center.x - center.x);
        float travel_y = abs(previous_center.y - center.y);
        float distance = sqrt((travel_x * travel_x) + (travel_y * travel_y)) * mpp;
        previous_center = center;
        trace[pos] = center;
        myfile.open("data.txt", ios::app);
        myfile << center << "\n";
        myfile.close();

        //Plots lines depending on colour selected
        for (int i = 0; i < pos; i++) {
          
          if (i > blue) {
            line(frame, trace[i], trace[i + 1], CV_RGB(0, 0, 255), 1.8, LINE_8, 0);
          }
          else if (i > green) {
            line(frame, trace[i], trace[i + 1], CV_RGB(0, 255, 0), 1.8, LINE_8, 0);
          }
          else {
            line(frame, trace[i], trace[i + 1], CV_RGB(255, 0, 0), 1.8, LINE_8, 0);
          }
        }
        pos += 1;

      }
      video.write(frame);
      imshow("Frame", frame);
      if (waitKey(20) == ESC_key) {
        imwrite("C:/Users/Callum/Documents/RoboNet/tracking/traces/Final.png", frame);
        cap.release();
        video.release();
        break;
      }
    }
  }
}