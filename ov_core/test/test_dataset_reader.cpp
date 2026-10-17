/* SPDX-License-Identifier: GPL-3.0-or-later */
#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "utils/dataset_reader.h"

TEST(DatasetReader, GroundTruthLookupPicksTheClosestEarlierOnTies) {
  std::map<double, Eigen::Matrix<double, 17, 1>> states;
  for (int i = 0; i < 100; i++) {
    Eigen::Matrix<double, 17, 1> state = Eigen::Matrix<double, 17, 1>::Zero();
    state(0) = 100.0 + 0.0625 * i; // exactly representable, so ties are exact
    state(4) = 1; // valid quaternion in the ETH [p,q,...] ordering
    states[state(0)] = state;
  }
  std::mt19937 random(3);
  std::uniform_real_distribution<double> times(99.9, 106.3);
  for (int trial = 0; trial < 1000; trial++) {
    const double query = times(random);
    // Reference: the original linear scan
    double expected = INFINITY;
    for (const auto &item : states) {
      if (std::abs(item.first - query) < std::abs(expected - query))
        expected = item.first;
    }
    Eigen::Matrix<double, 17, 1> found;
    const bool ok = ov_core::DatasetReader::get_gt_state(query, found, states);
    EXPECT_EQ(ok, std::abs(expected - query) < 0.10);
    if (ok)
      EXPECT_EQ(found(0), expected);
  }
  Eigen::Matrix<double, 17, 1> found;
  ASSERT_TRUE(ov_core::DatasetReader::get_gt_state(100.03125, found, states));
  EXPECT_EQ(found(0), 100.0); // equidistant from 100.0 and 100.0625
}
