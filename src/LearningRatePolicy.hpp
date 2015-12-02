
#pragma once

#include <memory>

using namespace std;


class LearningRatePolicy {
public:

  LearningRatePolicy(double initial, double decay);
  virtual ~LearningRatePolicy();

  double getLearningRate(void) const;
  void nextIter(void);
  void handleOvershoot(void);

private:
  class LearningRatePolicyImpl;
  unique_ptr<LearningRatePolicyImpl> impl;
};
