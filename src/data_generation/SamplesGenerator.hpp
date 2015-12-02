
#pragma once

#include "../common/SamplePoint.hpp"

class SamplesGenerator {
public:
  virtual ~SamplesGenerator() = default;

  virtual std::vector<SamplePoint> generate(unsigned num) const = 0;

};
