#include "neural.h"
#include "clbp/Net.h"

#include "iir1/Iir.h"

#include <vector>
#include <string>
#include <initializer_list>
#include <torch/torch.h>
#include <memory>
#include <boost/circular_buffer.hpp>
#include <opencv2/opencv.hpp>
#include "cvui.h"

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

    return torch::sigmoid(layers.back()->forward(x).sub(0.5));
  }


  //torch::nn::Linear layer1;
  //torch::nn::Linear layer2;
  std::vector<torch::nn::Linear> layers;
};


// If we use the IIR filters to create correlation with the error:
std::vector<std::array<Iir::Butterworth::LowPass<1>, 10>> lowpassFilters;
bool useFilters = false;

static void initialize_filters(int numInputs, float sampleRate)
{
  // Initialize the lowpass filters
  std::array<float, 10> bankCutoffFreqs = {1, 2, 3, 4, 5, 6, 7, 9, 11, 15};

  lowpassFilters.resize(numInputs);
  for (auto &bank : lowpassFilters) {
    for (int j = 0; j < 10; ++j) {
      bank[j].setup(sampleRate, bankCutoffFreqs[j]);
    }
  }
}


std::unique_ptr<RoboNet> net;
std::unique_ptr<torch::optim::SGD> optimizer;

boost::circular_buffer<std::vector<float>> previous_nn_outs(25);

void initialize_net(int numInputLayers, bool useFilters, float sampleRate)
{
  //torch::manual_seed(1);
  //RoboNet net;
  net = std::make_unique<RoboNet>(std::initializer_list<int>{numInputLayers, 12, 1});
  net->to(torch::kCPU);
  optimizer = std::make_unique<torch::optim::SGD>(net->parameters(), torch::optim::SGDOptions(0.01) );
}

// Error in range -1, 1. [-1, 0] wants the robot to turn left.
double run_nn(cv::Mat &statFrame, std::vector<float>& in, double error)
{
  torch::Tensor input = torch::from_blob(in.data(), in.size(), torch::kFloat32);

  auto output = net->forward(input);

  previous_nn_outs.push_back(in);

  // Output 0 - 0.5 turns left
  // Output 0.5 - 1 turns right

  float result = output.data<float>()[0];

  if (!previous_nn_outs.full()) return result;


  // Circular buffer is full. We can get the earlier inputs, calculate tensor(output)
  // and do backprop on that with the current error.





  auto old_inputs = previous_nn_outs.front();
  auto old_in_tensor = torch::from_blob(old_inputs.data(), old_inputs.size());
  auto old_output = net->forward(old_in_tensor).detach(); // We want to stop the graph at the output layer


  //std::cout << "dfsfs is " << res << " " << result << std::endl;
  cvui::printf(statFrame, 10, 10, "Old tensor output : %f", old_output.data<float>()[0]);
  cvui::printf(statFrame, 10, 30, "Current error :     %f", error);



  old_output.data<float>()[0] = error;
  old_output.backward();
  //torch::Tensor errorT = torch::from_blob(&res, 1, torch::kFloat32);


  //auto loss = torch::l1_loss(old_output, errorT);
  //auto loss = old_output + errorT;

  //loss.data<float>()[0] = res;

  if (error == 0) return result;

  //std::cout << "error is: " << loss.item().toFloat() << std::endl;



  //output.sub(leadError).backward();
  optimizer->step();

  return result;
}



std::unique_ptr<Net> samanet;



void initialize_samanet(int numInputLayers, bool useFilters, float sampleRate)
{
  ::useFilters = useFilters;
  if (useFilters)
    numInputLayers *= 10;

  int nNeurons[] = {numInputLayers, 12, 1};
  samanet = std::make_unique<Net>(3, nNeurons, numInputLayers);
  samanet->initWeights(Neuron::W_RANDOM, Neuron::B_NONE);
  samanet->setLearningRate(0.01);

  if (useFilters)
    initialize_filters(numInputLayers, sampleRate);
}

boost::circular_buffer<std::vector<float>> old_inputs(25);


double run_samanet(cv::Mat &statFrame, std::vector<float> &predictorDeltas, double error)
{
  old_inputs.push_back(predictorDeltas);

  std::vector<double> networkInputs;

  if (useFilters) {
    networkInputs.reserve(predictorDeltas.size() * 10);
    for (int j = 0; j < predictorDeltas.size(); ++j) {
      float sample = predictorDeltas[j];
      for (auto &filt : lowpassFilters[j]) {
        networkInputs.push_back(filt.filter(sample));
      }
    }
    samanet->setInputs(networkInputs.data());
    samanet->propInputs();
    samanet->setError(error);
    samanet->propError();
    samanet->updateWeights();
    samanet->saveWeights();
    samanet->getWeightDistance();
    return samanet->getOutput(0);
  }
  throw 1;

#if 0
  // Need a pointer to double.
  predictorDeltasDouble.resize(predictorDeltas.size());

  std::copy(predictorDeltas.begin(), predictorDeltas.end(), predictorDeltasDouble.begin());

  samanet->setInputs(predictorDeltasDouble.data());

  samanet->propInputs();

  auto net_out = samanet->getOutput(0);

  if (old_inputs.full()) {
    /* FIXME: I am not sure why the error is the following. It is in Sama's original
     algorithm, but I can't see how it doesn't saturate a sigmoid */
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
#endif
}