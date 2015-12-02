

#pragma once

#include <iosfwd>
#include <vector>

class VectorN final {
public:
  std::vector<double> vals;

  VectorN(unsigned n);
  VectorN(const std::vector<double> &vals);

  VectorN& normalise(void);
  VectorN normalised(void) const;

  double dotProduct(const VectorN &v) const;

  double length() const;
  double length2() const;

  VectorN operator+(const VectorN &v) const;
  VectorN operator-(const VectorN &v) const;
  VectorN operator*(double s) const;
  VectorN operator/(double s) const;

  VectorN& operator+=(const VectorN &v);
  VectorN& operator-=(const VectorN &v);
  VectorN& operator*=(double s);
  VectorN& operator/=(double s);
};

std::ostream& operator<<(std::ostream& stream, const VectorN& v);
const VectorN operator*(double s, const VectorN &v);
