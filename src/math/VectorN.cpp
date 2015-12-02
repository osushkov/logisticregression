
#include "VectorN.hpp"
#include <cmath>
#include <cassert>
#include <ostream>


VectorN::VectorN(unsigned n) : vals(n, 0.0) {}
VectorN::VectorN(const std::vector<double> &vals) : vals(vals) {}

VectorN& VectorN::normalise(void) {
  double il = 1.0 / length();
  for (auto& v : this->vals) {
    v *= il;
  }

  return *this;
}

VectorN VectorN::normalised(void) const {
  VectorN result(vals.size());
  double il = 1.0 / length();
  for (unsigned i = 0; i < vals.size(); i++) {
    result.vals[i] = vals[i] * il;
  }

  return result;
}

double VectorN::dotProduct(const VectorN &v) const {
  assert(this->vals.size() == v.vals.size());

  double r = 0.0;
  for (unsigned i = 0; i < vals.size(); i++) {
    r += vals[i] * v.vals[i];
  }
  return r;
}

double VectorN::length() const {
  return sqrt(length2());
}

double VectorN::length2() const {
  return dotProduct(*this);
}

VectorN VectorN::operator+(const VectorN &v) const {
  assert(this->vals.size() == v.vals.size());

  VectorN result(vals.size());
  for (unsigned i = 0; i < vals.size(); i++) {
    result.vals[i] = vals[i] + v.vals[i];
  }
  return result;
}

VectorN VectorN::operator-(const VectorN &v) const {
  assert(this->vals.size() == v.vals.size());

  VectorN result(vals.size());
  for (unsigned i = 0; i < vals.size(); i++) {
    result.vals[i] = vals[i] - v.vals[i];
  }
  return result;
}

VectorN VectorN::operator*(double s) const {
  VectorN result(vals.size());
  for (unsigned i = 0; i < vals.size(); i++) {
    result.vals[i] = vals[i] * s;
  }
  return result;
}

VectorN VectorN::operator/(double s) const {
  VectorN result(vals.size());
  for (unsigned i = 0; i < vals.size(); i++) {
    result.vals[i] = vals[i] / s;
  }
  return result;
}

VectorN& VectorN::operator+=(const VectorN &v) {
  assert(this->vals.size() == v.vals.size());

  for (unsigned i = 0; i < vals.size(); i++) {
    vals[i] += v.vals[i];
  }

  return *this;
}

VectorN& VectorN::operator-=(const VectorN &v) {
  assert(this->vals.size() == v.vals.size());

  for (unsigned i = 0; i < vals.size(); i++) {
    vals[i] -= v.vals[i];
  }

  return *this;
}

VectorN& VectorN::operator*=(double s) {
  for (unsigned i = 0; i < vals.size(); i++) {
    vals[i] *= s;
  }

  return *this;
}

VectorN& VectorN::operator/=(double s) {
  for (unsigned i = 0; i < vals.size(); i++) {
    vals[i] /= s;
  }

  return *this;
}

const VectorN operator*(double s, const VectorN &v) {
  VectorN result(v.vals.size());
  for (unsigned i = 0; i < v.vals.size(); i++) {
    result.vals[i] = v.vals[i] * s;
  }
  return result;
}

std::ostream& operator<<(std::ostream& stream, const VectorN& v) {
  stream << "V[" << v.vals.size() << "]";

  stream << "(";
  for (unsigned i = 0; i < v.vals.size(); i++) {
    if (i != 0) {
      stream << ",";
    }
    stream << v.vals[i];
  }
  stream << ")";
  return stream;
}
