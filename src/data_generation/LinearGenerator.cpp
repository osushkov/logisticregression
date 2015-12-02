
#include "LinearGenerator.hpp"
#include <cassert>


class LinearGenerator::LinearGeneratorImpl {
  unique_ptr<VectorN> coefficients;
  unique_ptr<vector<Range>> featureRange;
  unique_ptr<RandomDistribution> noise;

public:

  LinearGeneratorImpl(
      unique_ptr<VectorN> coefficients,
      unique_ptr<vector<Range>> featureRange,
      unique_ptr<RandomDistribution> noise) :
        coefficients(move(coefficients)),
        featureRange(move(featureRange)),
        noise(move(noise)) {
    assert(this->featureRange->size() == this->coefficients->vals.size());
  }

  vector<SamplePoint> generate(unsigned num) const {
    vector<SamplePoint> result;
    for (unsigned i = 0; i < num; i++) {
      result.push_back(generateSample());
    }
    return result;
  }

private:

  SamplePoint generateSample(void) const {
    VectorN featureVals {(unsigned) featureRange->size()};
    for (unsigned i = 0; i < featureRange->size(); i++) {
      featureVals.vals[i] = featureRange->at(i).randomPoint();
    }

    double fval = coefficients->dotProduct(featureVals) + noise->sample();
    return SamplePoint {featureVals, fval};
  }

};

LinearGenerator::LinearGenerator(
    unique_ptr<VectorN> coefficients,
    unique_ptr<vector<Range>> featureRange,
    unique_ptr<RandomDistribution> noise) :
  impl(new LinearGeneratorImpl {move(coefficients), move(featureRange), move(noise)}) {}

LinearGenerator::~LinearGenerator() = default;

vector<SamplePoint> LinearGenerator::generate(unsigned num) const {
  return impl->generate(num);
}
