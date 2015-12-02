
#pragma once

#include <cassert>
#include <cstdlib>

struct Range {
    double min, max;

    Range(double min, double max) : min(min), max(max) {
      assert(min <= max);
    }

    Range(double fixed) : Range(fixed, fixed) {}

    double randomPoint(void) const {
      return min + (max - min) * rand() / (double) RAND_MAX;
    }
};
