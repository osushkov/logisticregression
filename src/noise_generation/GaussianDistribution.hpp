
#pragma once

#include "RandomDistribution.hpp"

class GaussianDistribution : public RandomDistribution {

  const double stdDev;

public:
  // Will generate random varialbes with the given standard deviation, with a mean of 0.
  GaussianDistribution(double stdDev);

  double sample(void) const override;
};
