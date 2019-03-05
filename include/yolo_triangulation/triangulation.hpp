/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2009, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef __OPENCV_SFM_TRIANGULATION_HPP__
#define __OPENCV_SFM_TRIANGULATION_HPP__

#include <opencv2/core.hpp>

namespace cv
{
namespace sfm
{

//! @addtogroup projection
//! @{

/** @brief Converts point coordinates from homogeneous to euclidean pixel coordinates. E.g., ((x,y,z)->(x/z, y/z))
  @param src Input vector of N-dimensional points.
  @param dst Output vector of N-1-dimensional points.
*/
CV_EXPORTS_W
void
homogeneousToEuclidean(InputArray src, OutputArray dst);

//! @addtogroup triangulation
//! @{

/** @brief Reconstructs bunch of points by triangulation.
  @param points2d Input vector of vectors of 2d points (the inner vector is per image). Has to be 2 X N.
  @param projection_matrices Input vector with 3x4 projections matrices of each image.
  @param points3d Output array with computed 3d points. Is 3 x N.

  Triangulates the 3d position of 2d correspondences between several images.
  Reference: Internally it uses DLT method @cite HartleyZ00 12.2 pag.312
*/
CV_EXPORTS_W

void
triangulateNViews(const Mat_<double> &x, const std::vector<Matx34d> &Ps, Vec3d &X);

void
triangulatePoints(InputArrayOfArrays points2d, InputArrayOfArrays projection_matrices,
                  OutputArray points3d);

//! @} sfm

} /* namespace sfm */
} /* namespace cv */

#endif

/* End of file. */
