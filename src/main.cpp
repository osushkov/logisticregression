
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


VectorN getErrorGradients(const vector<SamplePoint> &samples, const VectorN &curCoeff) {
  VectorN result {(unsigned) curCoeff.vals.size()};

  for (unsigned i = 0; i < result.vals.size(); i++) {
    // compute d'err/d'theta(i)

    double derr = 0.0;
    for (auto& s : samples) {
      double hx = curCoeff.dotProduct(s.features);
      derr += 2.0 * s.features.vals[i] * (hx - s.value);
    }

    result.vals[i] = derr / samples.size();
  }

  return result;
}

double getError(const vector<SamplePoint> &samples, const VectorN &curCoeff) {
  double errSum = 0.0;

  for (auto& s : samples) {
    double hx = curCoeff.dotProduct(s.features);
    errSum += (hx - s.value) * (hx - s.value);
  }

  return errSum / samples.size();
}

VectorN performGradientDescent(const vector<SamplePoint> &samples, const VectorN &startPoint) {
  const unsigned NUM_ITER = 1000;
  double initial = 0.01;
  double final = 0.00001;
  double decay = exp(log(final / initial) / NUM_ITER);

  LearningRatePolicy lrp(initial, decay);

  VectorN cur(startPoint);
  double prevError = getError(samples, cur);

  for (unsigned i = 0; i < NUM_ITER; i++) {
    VectorN de = getErrorGradients(samples, cur);
    cur -= de * lrp.getLearningRate();

    double curError = getError(samples, cur);
    cout << i << "\t" << curError << endl;
    if (curError > prevError) {
      lrp.handleOvershoot();
    } else {
      lrp.nextIter();
    }
  }

  return cur;
}

int main() {
  auto noise = unique_ptr<RandomDistribution> {new GaussianDistribution {10.0}};
  auto coeff = unique_ptr<VectorN> {new VectorN {{1.0, 50.0, 10.0, -20, 0.1}}};
  auto featureRange = unique_ptr<vector<Range>> {new vector<Range> {}};
  featureRange->push_back(Range(1.0));

  // feature ranges should be small and around 0, otherwise gradient descent sucks
  featureRange->push_back(Range(-2.0, 2.0));
  featureRange->push_back(Range(-2.0, 2.0));
  featureRange->push_back(Range(-2.0, 2.0));
  featureRange->push_back(Range(-2.0, 2.0));

  auto sampleGenerator = unique_ptr<SamplesGenerator> {
      new LinearGenerator {move(coeff), move(featureRange), move(noise)}};

  vector<SamplePoint> samples = sampleGenerator->generate(1000);
  VectorN r = performGradientDescent(samples, VectorN {{0.5, 1.5, 1.0, 0.1, -0.1}});
  cout << r << endl;

	return 0;
}
