#include "neural.h"

#include <vector>
#include <string>
#include <initializer_list>
#include <torch/torch.h>
#include <memory>
struct RoboNet : public torch::nn::Module
{
  /*
  RoboNet(std::initializer_list<int> layer_neurons)
  {
    layers.reserve(layer_neurons.size());
    for (int j = 1; j < layer_neurons.size(); ++j) {
      layers.emplace_back(*(layer_neurons.begin() + j - 1), *(layer_neurons.begin() + j));
      register_module(std::string("layer") + std::to_string(j), layers.back());
    }
  }
  */
  RoboNet() :
    torch::nn::Module(),
    layer1(register_module("layer1", torch::nn::Linear(12, 5))),
    layer2(5, 1)
  {
    //register_module("layer1", layer1);
    register_module("layer2", layer2);
  }

  torch::Tensor forward(torch::Tensor x) {
    x = torch::sigmoid(layer1->forward(x));
    x = torch::sigmoid(layer2->forward(x));
    return x;
  }
  /*
  torch::Tensor forward(torch::Tensor x) {
    for (auto& l : layers) {
      x = torch::sigmoid(l->forward(x));
    }

    return x;
  }
  */

  torch::nn::Linear layer1;
  torch::nn::Linear layer2;
  //std::vector<torch::nn::Linear> layers;
};


std::unique_ptr<RoboNet> net;
std::unique_ptr<torch::optim::Adam> optimizer;


void initialize_net()
{
  RoboNet net;
  //net = std::make_unique<RoboNet>();
  //net->to(torch::kCPU);
  //optimizer = std::make_unique<torch::optim::Adam>(net->parameters(), torch::optim::AdamOptions(0.001));
}



double run_nn(std::vector<double>& in, double error)
{
  auto input = torch::from_blob(in.data(), in.size(), torch::kFloat64);

  auto output = net->forward(input);

  double leadError = error;

  std::cout << "fml\n";

  double result = output.data<double>()[0];

  std::cout << "fml2\n";


  output.sub(leadError).backward();
  optimizer->step();

  return result;
}