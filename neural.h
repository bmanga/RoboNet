#pragma once
#include <vector>

namespace cv {
class Mat;
}
void initialize_net();
double run_nn(cv::Mat &statFrame, std::vector<float>& in, double error);

void initialize_samanet();
double run_samanet(std::vector<float>& in, double error);