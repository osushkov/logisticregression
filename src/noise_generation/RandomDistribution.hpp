
#pragma once

class RandomDistribution {
public:
  virtual ~RandomDistribution() = default;

  virtual double sample(void) const = 0;
};
