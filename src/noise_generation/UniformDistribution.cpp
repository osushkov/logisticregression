
#include "UniformDistribution.hpp"
#include <cassert>
#include <cstdlib>


UniformDistribution::UniformDistribution(double range) : range(range) {
  assert(range >= 0.0);
}

double UniformDistribution::sample(void) const {
  return 2.0 * range * (std::rand() / (double) RAND_MAX) - range;
}
