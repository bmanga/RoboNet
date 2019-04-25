#include "neural.h"
#include "clbp/Net.h"

#include <vector>
#include <string>
#include <initializer_list>
#include <torch/torch.h>
#include <memory>
#include <boost/circular_buffer.hpp>

struct RoboNet : public torch::nn::Module
{

  RoboNet(std::initializer_list<int> layer_neurons)
  {
    layers.reserve(layer_neurons.size());
    for (int j = 1; j < layer_neurons.size(); ++j) {
      layers.emplace_back(*(layer_neurons.begin() + j - 1), *(layer_neurons.begin() + j));
      register_module(std::string("layer") + std::to_string(j), layers.back());
    }
  }
  /*
  RoboNet() :
    torch::nn::Module(),
    layer1(48, 12),
    layer2(12, 1)
  {
    register_module("layer1", layer1);
    register_module("layer2", layer2);
  }

  torch::Tensor forward(torch::Tensor x) {
    x = torch::sigmoid(layer1->forward(x));
    x = torch::sigmoid(layer2->forward(x));
    return x;
  }
  */
  torch::Tensor forward(torch::Tensor x) {
    for (auto it = layers.begin(); it != layers.end() - 1; ++it) {
      x = torch::relu((*it)->forward(x));
    }

    return torch::sigmoid(layers.back()->forward(x));
  }


  //torch::nn::Linear layer1;
  //torch::nn::Linear layer2;
  std::vector<torch::nn::Linear> layers;
};


std::unique_ptr<RoboNet> net;
std::unique_ptr<torch::optim::Adam> optimizer;

boost::circular_buffer<std::vector<float>> previous_nn_outs(25);

void initialize_net()
{
  //torch::manual_seed(1);
  //RoboNet net;
  net = std::make_unique<RoboNet>(std::initializer_list<int>{6, 12, 1});
  net->to(torch::kCPU);
  optimizer = std::make_unique<torch::optim::Adam>(net->parameters(), torch::optim::AdamOptions(0.001) );
}

// Error in range -1, 1. [-1, 0] wants the robot to turn left.
double run_nn(std::vector<float>& in, double error)
{
  float error_center = 0.5;
  error = error / 1.3;
  torch::Tensor input = torch::from_blob(in.data(), in.size(), torch::kFloat32);

  auto output = net->forward(input);

  previous_nn_outs.push_back(in);

  // Output 0 - 0.5 turns left
  // Output 0.5 - 1 turns right

  float result = output.data<float>()[0];

  if (!previous_nn_outs.full()) return result - error_center;


  // Circular buffer is full. We can get the earlier inputs, calculate tensor(output)
  // and do backprop on that with the current error.





  auto old_inputs = previous_nn_outs.front();
  auto old_in_tensor = torch::from_blob(old_inputs.data(), old_inputs.size());
  auto old_output = net->forward(old_in_tensor);

  float res = error_center + error;

  //std::cout << "dfsfs is " << res << " " << result << std::endl;



  torch::Tensor errorT = torch::from_blob(&res, 1, torch::kFloat32);

  std::cout << "- tensors -: ";
  std::cout << output.data<float>()[0] << " " << old_output.data<float>()[0] << " " << errorT.data<float>()[0] << std::endl;

  auto loss = torch::l1_loss(old_output, errorT);
  //auto loss = old_output + errorT;

  loss.data<float>()[0] = res;

  if (loss.item().toFloat() == 0) return result - error_center;

  //std::cout << "error is: " << loss.item().toFloat() << std::endl;
  loss.backward();

  //output.sub(leadError).backward();
  optimizer->step();

  return result - error_center;
}

std::unique_ptr<Net> samanet;
void initialize_samanet()
{
  int nNeurons[] = {6, 12, 1};
  samanet = std::make_unique<Net>(3, nNeurons, 6);
  samanet->initWeights(Neuron::W_ONES, Neuron::B_NONE);

  samanet->setLearningRate(0.01);

}

boost::circular_buffer<std::vector<float>> old_inputs(25);

double run_samanet(std::vector<float> &predictorDeltas, double error)
{
  error /= 1.3;

  old_inputs.push_back(predictorDeltas);

  // Need a pointer to double.
  std::vector<double> predictorDeltasDouble;
  predictorDeltasDouble.resize(predictorDeltas.size());

  std::copy(predictorDeltas.begin(), predictorDeltas.end(), predictorDeltasDouble.begin());

  samanet->setInputs(predictorDeltasDouble.data());

  samanet->propInputs();

  auto net_out = samanet->getOutput(0);

  if (old_inputs.full()) {
    /* FIXME: I am not sure why the error is the following. It is in Sama's original
     algorithm, but I can't see how it relates to the output of the network. */
    double leadError = 5 * error;

    std::copy(old_inputs.front().begin(), old_inputs.front().end(), predictorDeltasDouble.begin());
    samanet->setInputs(predictorDeltasDouble.data());
    samanet->propInputs();
    samanet->setError(-leadError);
    samanet->propError();
    samanet->updateWeights();
  }
  //need to do weight change first
  //net.saveWeights();

  return net_out;
}