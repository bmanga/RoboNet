#pragma once
#include <vector>

void initialize_net();
double run_nn(std::vector<float>& in, double error);

void initialize_samanet();
double run_samanet(std::vector<float>& in, double error);