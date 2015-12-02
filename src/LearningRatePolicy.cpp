
#include "LearningRatePolicy.hpp"
#include <cassert>


class LearningRatePolicy::LearningRatePolicyImpl {

  double curRate;
  const double decay;

public:
  LearningRatePolicyImpl(double initial, double decay) :
      curRate(initial), decay(decay) {
    assert(initial > 0.0);
    assert(decay > 0.0 && decay < 1.0);
  }

  double getLearningRate(void) const {
    return curRate;
  }

  void nextIter(void) {
    curRate *= decay;
  }

  void handleOvershoot(void) {
    curRate *= 0.5;
  }

};


LearningRatePolicy::LearningRatePolicy(double initial, double decay) :
    impl(new LearningRatePolicyImpl(initial, decay)) {}

LearningRatePolicy::~LearningRatePolicy() = default;

double LearningRatePolicy::getLearningRate(void) const {
  return impl->getLearningRate();
}

void LearningRatePolicy::nextIter(void) {
  impl->nextIter();
}

void LearningRatePolicy::handleOvershoot(void) {
  impl->handleOvershoot();
}
