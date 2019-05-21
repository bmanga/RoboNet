#include "neural.h"
#include "clbp/Net.h"

#include "iir1/Iir.h"

#include <vector>
#include <string>
#include <initializer_list>
#include <memory>
#include <chrono>
#include <fstream>
#include <boost/circular_buffer.hpp>
#include <opencv2/opencv.hpp>
#include "cvui.h"

#include "bandpass.h"


// If we use the IIR filters to create correlation with the error:

//std::vector<std::array<Iir::Butterworth::LowPass<1>, 5>> lowpassFilters;

std::vector<std::array<Bandpass, 5>> bandpassFilters;

bool useFilters = false;

#if 0
static void initialize_filters(int numInputs, float sampleRate)
{
  // Initialize the lowpass filters
  std::array<float, 5> bankCutoffFreqs = {1, 3, 5, 7, 11};//{1, 2, 3, 4, 5, 6, 7, 9, 11, 15};

  lowpassFilters.resize(numInputs);
  for (auto &bank : lowpassFilters) {
    for (int j = 0; j < 5; ++j) {
      bank[j].setup(sampleRate, bankCutoffFreqs[j]);
    }
  }
}
#else
static void initialize_filters(int numInputs, float sampleRate)
{
  bandpassFilters.resize(numInputs);
  double fs = 1;
  double fmin = fs / 39;
  double fmax = fs / 9;
  double df = (fmax - fmin) / 4.0;
  for (auto &bank : bandpassFilters) {
    double f = fmin;
    for (auto &filt : bank) {
      filt.setParameters(f, 0.51);
      f += df;
    }
  }
}
#endif

boost::circular_buffer<std::vector<float>> previous_nn_outs(25);

std::unique_ptr<Net> samanet;

void initialize_samanet(int numInputLayers, bool useFilters, float sampleRate)
{
  ::useFilters = useFilters;
  if (useFilters)
    numInputLayers *= 5;

  int nNeurons[] = {16, 8, 1};
  samanet = std::make_unique<Net>(3, nNeurons, numInputLayers);
  samanet->initWeights(Neuron::W_RANDOM, Neuron::B_NONE);
  samanet->setLearningRate(0.05 );

  if (useFilters)
    initialize_filters(numInputLayers, sampleRate);
}

boost::circular_buffer<std::vector<float>> old_inputs(25);


std::ofstream weightDistancesfs ("weight_distances.csv");

std::ofstream filterout("filterouts.csv");
std::ofstream unfilteredout("unfilteredouts.csv");

double run_samanet(cv::Mat &statFrame, std::vector<float> &predictorDeltas, double error)
{


  using namespace std::chrono;
  milliseconds ms = duration_cast< milliseconds >(
    system_clock::now().time_since_epoch()
  );

  old_inputs.push_back(predictorDeltas);

  std::vector<double> networkInputs;

  if (useFilters) {
    filterout << "\n" << ms.count();
    unfilteredout << "\n" << ms.count();
    networkInputs.reserve(predictorDeltas.size() * 5);
    for (int j = 0; j < predictorDeltas.size(); ++j) {
      float sample = predictorDeltas[j];
      for (auto &filt : bandpassFilters[j]) {
        auto filtered = filt.filter(sample);
        networkInputs.push_back(filtered);
        if (j == 0) {
          unfilteredout << "," << sample;
          filterout << "," << filtered;
        }
      }
    }
    samanet->setInputs(networkInputs.data());
    samanet->propInputs();
    samanet->setError(error);
    samanet->propError();
    samanet->updateWeights();
    //samanet->saveWeights();
    //samanet->getWeightDistance();

    weightDistancesfs << ms.count() << ","
                      << samanet->getWeightDistanceLayer(0) << ","
                      << samanet->getWeightDistanceLayer(1) << ","
                      << samanet->getWeightDistanceLayer(2) << "\n";
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
