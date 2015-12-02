
#include "noise_generation/UniformDistribution.hpp"
#include "noise_generation/GaussianDistribution.hpp"
#include "data_generation/SamplesGenerator.hpp"
#include "data_generation/LinearGenerator.hpp"
#include "math/VectorN.hpp"
#include "math/Range.hpp"
#include "common/SamplePoint.hpp"
#include "LearningRatePolicy.hpp"
#include <iostream>
#include <vector>
#include <memory>
#include <cmath>

using namespace std;


double logisticFunc(double x) {
  return 1.0 / (1.0 + exp(-x));
}

VectorN getErrorGradients(const vector<SamplePoint> &samples, const VectorN &curCoeff) {
  VectorN result {(unsigned) curCoeff.vals.size()};

  for (unsigned i = 0; i < result.vals.size(); i++) {
    // compute d'err/d'theta(i)

    double derr = 0.0;
    for (auto& s : samples) {
      double hx = logisticFunc(curCoeff.dotProduct(s.features));
      derr += s.features.vals[i] * (hx - s.category);
    }

    result.vals[i] = derr / samples.size();
  }

  return result;
}

double getError(const vector<SamplePoint> &samples, const VectorN &curCoeff) {
  double errSum = 0.0;

  for (auto& s : samples) {
    double hx = logisticFunc(curCoeff.dotProduct(s.features));
    errSum += (hx - s.category) * (hx - s.category);
  }

  return errSum / samples.size();
}

VectorN performGradientDescent(const vector<SamplePoint> &samples, const VectorN &startPoint) {
    // return startPoint;

  const unsigned NUM_ITER = 100000;
  double initial = 0.01;
  double final = 0.000001;
  double decay = exp(log(final / initial) / NUM_ITER);

  LearningRatePolicy lrp(initial, decay);

  VectorN cur(startPoint);
  double prevError = getError(samples, cur);

  for (unsigned i = 0; i < NUM_ITER; i++) {
    VectorN de = getErrorGradients(samples, cur);
    cur -= de * lrp.getLearningRate();

    double curError = getError(samples, cur);
    // cout << i << "\t" << curError << endl;
    if (curError > prevError) {
      lrp.handleOvershoot();
    } else {
      lrp.nextIter();
    }
  }

  return cur;
}

VectorN learnModel(SamplesGenerator *samplesGenerator) {
  vector<SamplePoint> samples = samplesGenerator->generate(10000);
  return performGradientDescent(samples, VectorN {{0.1, 0.1, -0.1, 0.1, -0.1}});
}

double evaluatePerformance(const VectorN &learnedModel, SamplesGenerator *samplesGenerator) {
  vector<SamplePoint> validationSet = samplesGenerator->generate(1000);

  unsigned numCorrect = 0;
  unsigned numIncorrect = 0;

  for (auto& vs : validationSet) {
    double x = vs.features.dotProduct(learnedModel);
    double classification = logisticFunc(x) > 0.5 ? 1.0 : 0.0;
    bool isCorrect = (vs.category > 0.5) == (classification > 0.5);
    if (isCorrect) {
      numCorrect++;
    } else {
      numIncorrect++;
    }
  }

  return numCorrect / (double) (numCorrect + numIncorrect);
}

int main() {
  auto noise = unique_ptr<RandomDistribution> {new GaussianDistribution {5.0}};
  auto coeff = unique_ptr<VectorN> {new VectorN {{1.0, 50.0, 10.0, -20, 0.1}}};
  auto featureRange = unique_ptr<vector<Range>> {new vector<Range> {}};
  featureRange->push_back(Range(1.0)); // this is just the constant feature.

  // feature ranges should be small and around 0, otherwise gradient descent sucks
  featureRange->push_back(Range(-1.0, 1.0));
  featureRange->push_back(Range(-1.0, 1.0));
  featureRange->push_back(Range(-1.0, 1.0));
  featureRange->push_back(Range(-1.0, 1.0));

  auto sampleGenerator = unique_ptr<SamplesGenerator> {
      new LinearGenerator {move(coeff), move(featureRange), move(noise)}};


  VectorN learnedModel = learnModel(sampleGenerator.get());
  double perf = evaluatePerformance(learnedModel, sampleGenerator.get());

  cout << "performance: " << perf << endl;
  cout << "decision boundary: " << learnedModel << endl;

	return 0;
}
