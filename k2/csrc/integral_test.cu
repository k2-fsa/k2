/**
 * @brief Unittest for integral algorithm (this is really the application!)
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corp.   (Author: Daniel Povey)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <gtest/gtest.h>

#include <string>

#include "k2/csrc/integral.h"

namespace k2 {


TEST(ComputeIntegral, SinglePointAtOrigin) {

  // configuration with a single unit mass located at the origin.
  // should be the default (zero initialization) anyway...
  Configuration configuration;
  configuration.max_allowed_density = 1.0e+05;
  for (int i = 0; i < 3; i++)
    configuration.points[0].x[0] = 0;
  configuration.masses[0] = 1.0;

  ContextPtr c = GetCudaContext();
  double r = 10.0;  // cube radius (==half edge length)
  double integral_error;
  double integral = ComputeIntegral(c, configuration, r,
                                    1.0e-06,
                                    &integral_error);
  K2_LOG(INFO) << "For r = " << r << ", one mass at origin, integral = "
               << integral << " with error " << integral_error;
}  

}  // namespace k2
