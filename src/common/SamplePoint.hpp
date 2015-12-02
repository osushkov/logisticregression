
#pragma once

#include "../math/VectorN.hpp"
#include <ostream>


struct SamplePoint {
  VectorN features;
  double category; // 0 or 1

  SamplePoint(const VectorN &features, double category) :
    features(features), category(category) {}
};

inline std::ostream& operator<<(std::ostream& stream, const SamplePoint& v) {
  stream << v.category << " : " << v.features;
  return stream;
}
