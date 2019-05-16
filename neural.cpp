#include "neural.h"
#include "clbp/Net.h"

#include "iir1/Iir.h"

#include <vector>
#include <string>
#include <initializer_list>
#include <memory>
#include <boost/circular_buffer.hpp>
#include <opencv2/opencv.hpp>
#include "cvui.h"


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

boost::circular_buffer<std::vector<float>> previous_nn_outs(25);

std::unique_ptr<Net> samanet;

void initialize_samanet(int numInputLayers, bool useFilters, float sampleRate)
{
  ::useFilters = useFilters;
  if (useFilters)
    numInputLayers *= 10;

  int nNeurons[] = {numInputLayers, 12, 1};
  samanet = std::make_unique<Net>(3, nNeurons, numInputLayers);
  samanet->initWeights(Neuron::W_RANDOM, Neuron::B_NONE);
  samanet->setLearningRate(0.1);

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
    samanet->setError(-error);
    samanet->propError();
    samanet->updateWeights();
    //samanet->saveWeights();
    //samanet->getWeightDistance();
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

void dump_samanet()
{
  samanet->saveWeights();
}