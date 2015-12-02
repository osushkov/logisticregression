
#pragma once

#include "SamplesGenerator.hpp"
#include "../math/VectorN.hpp"
#include "../math/Range.hpp"
#include "../noise_generation/RandomDistribution.hpp"
#include <memory>
#include <vector>

using namespace std;

class LinearGenerator : public SamplesGenerator {
public:

  LinearGenerator(
      unique_ptr<VectorN> coefficients,
      unique_ptr<vector<Range>> featureRange,
      unique_ptr<RandomDistribution> noise);

  virtual ~LinearGenerator();

  vector<SamplePoint> generate(unsigned num) const override;

private:
  class LinearGeneratorImpl;
  unique_ptr<LinearGeneratorImpl> impl;
};
