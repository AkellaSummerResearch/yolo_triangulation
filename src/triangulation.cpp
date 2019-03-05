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

// #include "precomp.hpp"

// Eigen
#include <Eigen/Core>

// OpenCV
// #include <opencv2/core/eigen.hpp>
#include <yolo_triangulation/triangulation.hpp>
// #include <opencv2/sfm/projection.hpp>

// libmv headers
// #include "libmv/multiview/twoviewtriangulation.h"
// #include "libmv/multiview/fundamental.h"

using namespace cv;
using namespace std;

namespace cv
{
namespace sfm
{

template<typename T>
void
homogeneousToEuclidean(const Mat & _X, Mat & _x)
{
  int d = _X.rows - 1;

  const Mat_<T> & X_rows = _X.rowRange(0,d);
  const Mat_<T> h = _X.row(d);

  const T * h_ptr = h[0], *h_ptr_end = h_ptr + h.cols;
  const T * X_ptr = X_rows[0];
  T * x_ptr = _x.ptr<T>(0);
  for (; h_ptr != h_ptr_end; ++h_ptr, ++X_ptr, ++x_ptr)
  {
    const T * X_col_ptr = X_ptr;
    T * x_col_ptr = x_ptr, *x_col_ptr_end = x_col_ptr + d * _x.step1();
    for (; x_col_ptr != x_col_ptr_end; X_col_ptr+=X_rows.step1(), x_col_ptr+=_x.step1() )
      *x_col_ptr = (*X_col_ptr) / (*h_ptr);
  }
}

void
homogeneousToEuclidean(const InputArray _X, OutputArray _x)
{
  // src
  const Mat X = _X.getMat();

  // dst
   _x.create(X.rows-1, X.cols, X.type());
  Mat x = _x.getMat();

  // type
  if( X.depth() == CV_32F )
  {
    homogeneousToEuclidean<float>(X,x);
  }
  else
  {
    homogeneousToEuclidean<double>(X,x);
  }
}

// /** @brief Triangulates the a 3d position between two 2d correspondences, using the DLT.
//   @param xl Input vector with first 2d point.
//   @param xr Input vector with second 2d point.
//   @param Pl Input 3x4 first projection matrix.
//   @param Pr Input 3x4 second projection matrix.
//   @param objectPoint Output vector with computed 3d point.

//   Reference: @cite HartleyZ00 12.2 pag.312
//  */
// void
// triangulateDLT( const Vec2d &xl, const Vec2d &xr,
//                 const Matx34d &Pl, const Matx34d &Pr,
//                 Vec3d &point3d )
// {
//     Matx44d design;
//     for (int i = 0; i < 4; ++i)
//     {
//         design(0,i) = xl(0) * Pl(2,i) - Pl(0,i);
//         design(1,i) = xl(1) * Pl(2,i) - Pl(1,i);
//         design(2,i) = xr(0) * Pr(2,i) - Pr(0,i);
//         design(3,i) = xr(1) * Pr(2,i) - Pr(1,i);
//     }

//     Vec4d XHomogeneous;
//     cv::SVD::solveZ(design, XHomogeneous);

//     homogeneousToEuclidean(XHomogeneous, point3d);
// }


/** @brief Triangulates the 3d position of 2d correspondences between n images, using the DLT
 * @param x Input vectors of 2d points (the inner vector is per image). Has to be 2xN
 * @param Ps Input vector with 3x4 projections matrices of each image.
 * @param X Output vector with computed 3d point.

 * Reference: it is the standard DLT; for derivation see appendix of Keir's thesis
 */
void
triangulateNViews(const Mat_<double> &x, const std::vector<Matx34d> &Ps, Vec3d &X)
{
    CV_Assert(x.rows == 2);
    unsigned nviews = x.cols;
    CV_Assert(nviews == Ps.size());

    cv::Mat_<double> design = cv::Mat_<double>::zeros(3*nviews, 4 + nviews);
    for (unsigned i=0; i < nviews; ++i) {
        for(char jj=0; jj<3; ++jj)
            for(char ii=0; ii<4; ++ii)
                design(3*i+jj, ii) = -Ps[i](jj, ii);
        design(3*i + 0, 4 + i) = x(0, i);
        design(3*i + 1, 4 + i) = x(1, i);
        design(3*i + 2, 4 + i) = 1.0;
    }

    Mat X_and_alphas;
    cv::SVD::solveZ(design, X_and_alphas);
    homogeneousToEuclidean(X_and_alphas.rowRange(0, 4), X);
}



} /* namespace sfm */
} /* namespace cv */