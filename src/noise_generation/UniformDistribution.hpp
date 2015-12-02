
#pragma once

#include "RandomDistribution.hpp"
#include <memory>


class UniformDistribution : public RandomDistribution {

  const double range;

public:
  // Will generate random variables in the range [-range ... range]
  UniformDistribution(double range);

  double sample(void) const override;
};
