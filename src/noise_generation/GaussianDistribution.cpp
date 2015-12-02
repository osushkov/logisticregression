
#include "GaussianDistribution.hpp"
#include <cassert>
#include <cstdlib>
#include <cmath>


static float randInterval(double s, double e) {
  return s + (e - s) * (rand() / (double) RAND_MAX);
}

GaussianDistribution::GaussianDistribution(double stdDev) : stdDev(stdDev) {
  assert(stdDev >= 0.0);
}

double GaussianDistribution::sample(void) const {
  double x, y, r2;

  do{
    // choose x,y in uniform square (-1,-1) to (+1,+1)
    x = -1 + 2 * randInterval(0.0, 1.0);
    y = -1 + 2 * randInterval(0.0, 1.0);

    // see if it is in the unit circle
    r2 = x * x + y * y;
  } while (r2 > 1.0 || r2 == 0);

  // Box-Muller transform
  return stdDev * y * sqrt (-2.0 * log (r2) / r2);
}
