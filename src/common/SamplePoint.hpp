
#pragma once

#include "../math/VectorN.hpp"
#include <ostream>


struct SamplePoint {
  VectorN features;
  double value;

  SamplePoint(const VectorN &features, double value) :
    features(features), value(value) {}
};

inline std::ostream& operator<<(std::ostream& stream, const SamplePoint& v) {
  stream << v.value << " : " << v.features;
  return stream;
}
