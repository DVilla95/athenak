#ifndef PARTICLES_PARTICLES_HELPERS_HPP_
#define PARTICLES_PARTICLES_HELPERS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particles_helpers.hpp
//!  \brief
//!  Contains functions used by the particle pushers and module

#include "athena.hpp"
#include "particles.hpp"
#include "coordinates/cell_locations.hpp"

//----------------------------------------------------------------------------------------
//
//! \fn  void GetUpperAdmMetric
//  \brief
KOKKOS_INLINE_FUNCTION
void GetUpperAdmMetric( const Real inputMat[][4], Real outputMat[][3] ){
	for (int i1 = 0; i1 < 3; ++i1 ){ 
		for (int i2 = 0; i2 < 3; ++i2 ){ 
		outputMat[i1][i2] = inputMat[i1+1][i2+1] - inputMat[0][i2+1]*inputMat[i1+1][0]/inputMat[0][0];
		}
	}
}

//----------------------------------------------------------------------------------------
//! \fn  void ComputeDeterminant3
//  \brief Compute the determinant of a 3x3 matrix
KOKKOS_INLINE_FUNCTION
void ComputeDeterminant3( const Real inputMat[][3], Real &determinant ){

	determinant = inputMat[0][0]*inputMat[1][1]*inputMat[2][2] + inputMat[0][1]*inputMat[1][2]*inputMat[2][0]
		+ inputMat[1][0]*inputMat[2][1]*inputMat[0][2]
		- inputMat[2][0]*inputMat[1][1]*inputMat[0][2] - inputMat[1][0]*inputMat[0][1]*inputMat[2][2]
		- inputMat[2][1]*inputMat[1][2]*inputMat[1][1];

}

//----------------------------------------------------------------------------------------
//! \fn  void ComputeInverseMatrix3
//  \brief Compute the inverse of a 3x3 matrix
KOKKOS_INLINE_FUNCTION
void ComputeInverseMatrix3( const Real inputMat[][3], Real outputMat[][3] ){

	Real determinant = 0.0;
	ComputeDeterminant3( inputMat, determinant );

	// Transposition of Jacobian is skipped and indeces are inverted instead
	outputMat[0][0] = (inputMat[1][1]*inputMat[2][2] - inputMat[2][1]*inputMat[1][2])/determinant;
	outputMat[1][0] = -(inputMat[1][0]*inputMat[2][2] - inputMat[0][2]*inputMat[2][1])/determinant;
	outputMat[2][0] = (inputMat[1][1]*inputMat[1][2] - inputMat[0][2]*inputMat[1][1])/determinant;
	outputMat[0][1] = -(inputMat[1][0]*inputMat[2][2] - inputMat[1][2]*inputMat[2][0])/determinant;
	outputMat[1][1] = (inputMat[0][0]*inputMat[2][2] - inputMat[0][2]*inputMat[2][0])/determinant;
	outputMat[2][1] = -(inputMat[0][0]*inputMat[1][2] - inputMat[0][2]*inputMat[1][0])/determinant;
	outputMat[0][2] = (inputMat[1][0]*inputMat[2][1] - inputMat[1][1]*inputMat[2][0])/determinant;
	outputMat[1][2] = -(inputMat[0][0]*inputMat[2][1] - inputMat[0][1]*inputMat[2][0])/determinant;
	outputMat[2][2] = (inputMat[0][0]*inputMat[1][1] - inputMat[0][1]*inputMat[1][0])/determinant;

}

//----------------------------------------------------------------------------------------
//! \fn  void InterpolateFields
//  \brief Interpolate cell field to particle location,
KOKKOS_INLINE_FUNCTION
void InterpolateFields( const Real * prtcl_x, const DvceFaceFld4D<Real> &b0_, const DvceEdgeFld4D<Real> &e0_,
				const DualArray1D<RegionSize> &mbsize, const RegionIndcs &indcs, const int m,
			  Real * E, Real * B, bool &out_of_bounds ){

	int ip = (prtcl_x[0] - mbsize.d_view(m).x1min)/mbsize.d_view(m).dx1 + indcs.is;
	int jp = (prtcl_x[1] - mbsize.d_view(m).x2min)/mbsize.d_view(m).dx2 + indcs.js;
	int kp = (prtcl_x[2] - mbsize.d_view(m).x3min)/mbsize.d_view(m).dx3 + indcs.ks;
  // Sanity check: sometimes particles can make excessively large steps during NL iterations.
  // Returning the boolean as false allows to reset the iteration variables without crashing the whole code
  if (ip < 0 || jp < 0 || kp < 0
      || ip > (indcs.ie + indcs.ng) || jp > (indcs.je + indcs.ng) || kp > (indcs.ke + indcs.ng) ) {
    out_of_bounds = true;
    return;
  }
	Real &x1min = mbsize.d_view(m).x1min;
	Real &x2min = mbsize.d_view(m).x2min;
	Real &x3min = mbsize.d_view(m).x3min;
	Real &x1max = mbsize.d_view(m).x1max;
	Real &x2max = mbsize.d_view(m).x2max;
	Real &x3max = mbsize.d_view(m).x3max;
	Real x1v = LeftEdgeX(ip, indcs.nx1, x1min, x1max);
	Real x2v = LeftEdgeX(jp, indcs.nx2, x2min, x2max);
	Real x3v = LeftEdgeX(kp, indcs.nx3, x3min, x3max);
	Real Dx = (x1max - x1min)/indcs.nx1;
	Real Dy = (x2max - x2min)/indcs.nx2;
	Real Dz = (x3max - x3min)/indcs.nx3;
	// Interpolate Electric Field at new particle location x1, x2, x3

	E[0] = e0_.x1e(m, kp, jp, ip) + (prtcl_x[0] - x1v)*(e0_.x1e(m, kp, jp, ip+1) - e0_.x1e(m, kp, jp, ip))/Dx;
	E[0] += e0_.x1e(m, kp, jp, ip) + (prtcl_x[1] - x2v)*(e0_.x1e(m, kp, jp+1, ip) - e0_.x1e(m, kp, jp, ip))/Dy;
	E[0] += e0_.x1e(m, kp, jp, ip) + (prtcl_x[2] - x3v)*(e0_.x1e(m, kp+1, jp, ip) - e0_.x1e(m, kp, jp, ip))/Dz;
	E[1] = e0_.x2e(m, kp, jp, ip) + (prtcl_x[0] - x1v)*(e0_.x2e(m, kp, jp, ip+1) - e0_.x2e(m, kp, jp, ip))/Dx;
	E[1] += e0_.x2e(m, kp, jp, ip) + (prtcl_x[1] - x2v)*(e0_.x2e(m, kp, jp+1, ip) - e0_.x2e(m, kp, jp, ip))/Dy;
	E[1] += e0_.x2e(m, kp, jp, ip) + (prtcl_x[2] - x3v)*(e0_.x2e(m, kp+1, jp, ip) - e0_.x2e(m, kp, jp, ip))/Dz;
	E[2] = e0_.x3e(m, kp, jp, ip) + (prtcl_x[0] - x1v)*(e0_.x3e(m, kp, jp, ip+1) - e0_.x3e(m, kp, jp, ip))/Dx;
	E[2] += e0_.x3e(m, kp, jp, ip) + (prtcl_x[1] - x2v)*(e0_.x3e(m, kp, jp+1, ip) - e0_.x3e(m, kp, jp, ip))/Dy;
	E[2] += e0_.x3e(m, kp, jp, ip) + (prtcl_x[2] - x3v)*(e0_.x3e(m, kp+1, jp, ip) - e0_.x3e(m, kp, jp, ip))/Dz;

	// Interpolate Magnetic Field at new particle location x1, x2, x3
	B[0] = b0_.x1f(m, kp, jp, ip) + (prtcl_x[0] - x1v)*(b0_.x1f(m, kp, jp, ip+1) - b0_.x1f(m, kp, jp, ip))/Dx;
	B[0] += b0_.x1f(m, kp, jp, ip) + (prtcl_x[1] - x2v)*(b0_.x1f(m, kp, jp+1, ip) - b0_.x1f(m, kp, jp, ip))/Dy;
	B[0] += b0_.x1f(m, kp, jp, ip) + (prtcl_x[2] - x3v)*(b0_.x1f(m, kp+1, jp, ip) - b0_.x1f(m, kp, jp, ip))/Dz;
	B[1] = b0_.x2f(m, kp, jp, ip) + (prtcl_x[0] - x1v)*(b0_.x2f(m, kp, jp, ip+1) - b0_.x2f(m, kp, jp, ip))/Dx;
	B[1] += b0_.x2f(m, kp, jp, ip) + (prtcl_x[1] - x2v)*(b0_.x2f(m, kp, jp+1, ip) - b0_.x2f(m, kp, jp, ip))/Dy;
	B[1] += b0_.x2f(m, kp, jp, ip) + (prtcl_x[2] - x3v)*(b0_.x2f(m, kp+1, jp, ip) - b0_.x2f(m, kp, jp, ip))/Dz;
	B[2] = b0_.x3f(m, kp, jp, ip) + (prtcl_x[0] - x1v)*(b0_.x3f(m, kp, jp, ip+1) - b0_.x3f(m, kp, jp, ip))/Dx;
	B[2] += b0_.x3f(m, kp, jp, ip) + (prtcl_x[1] - x2v)*(b0_.x3f(m, kp, jp+1, ip) - b0_.x3f(m, kp, jp, ip))/Dy;
	B[2] += b0_.x3f(m, kp, jp, ip) + (prtcl_x[2] - x3v)*(b0_.x3f(m, kp+1, jp, ip) - b0_.x3f(m, kp, jp, ip))/Dz;
  for (int i = 0; i<3; ++i) {
    E[i] /= 3.0;
    B[i] /= 3.0;
  }

}

//----------------------------------------------------------------------------------------
//! \fn  void LUDecomposition
//  \brief Compute matrix to use as Lower and Upper triangular decomposition in matrix inversion
KOKKOS_INLINE_FUNCTION
void LUDecomposition( const int ndim, Real * LUMat, int * perm, bool &fail ){

    for (int i = 0; i<ndim; ++i)
      perm[i] = i;

    for (int k = 0; k<ndim; ++k) {
      int fixp = k;
      Real maxval = fabs(LUMat[perm[k]*ndim + k]);
      for (int i = k+1; i<ndim; ++i) {
        Real thisval = fabs(LUMat[perm[i]*ndim + k]);
        if (thisval > maxval) {
          maxval = thisval;
          fixp = i;
        }
      }

      if (maxval < Real(1e-14)) {
        fail = true;
        return;
      } 

      if (fixp != k) {
        Real tmp = perm[fixp];
        perm[fixp] = perm[k];
        perm[k] = tmp;
      }

      for (int i = k+1; i<ndim; ++i) {
        Real ratio = LUMat[perm[i]*ndim + k] / LUMat[perm[k]*ndim + k];
        LUMat[perm[i]*ndim + k] = ratio;
        for (int j = k+1; j<ndim; ++j)
          LUMat[perm[i]*ndim + j] -= ratio * LUMat[perm[k]*ndim + j];
      }
    }

    return;
}

//----------------------------------------------------------------------------------------
//! \fn  void FWDSubstitution
//  \brief First step of the inversion
KOKKOS_INLINE_FUNCTION
void FWDSubstitution(const int ndim, const Real * LUMat, const int * perm, const Real * idArr, Real * outArr) {

    for (int i = 0; i < ndim; ++i) {
        Real sum = 0.0;
        for (int j = 0; j < i; ++j)
          sum += LUMat[perm[i]*ndim + j] * outArr[j];
        
        outArr[i] = idArr[perm[i]] - sum;
    }

    return;
}

//----------------------------------------------------------------------------------------
//! \fn  void BWDSubstitution
//  \brief Second step of the inversion
KOKKOS_INLINE_FUNCTION
void BWDSubstitution(const int ndim, const Real * LUMat, const int * perm, const Real * inArr, Real * outArr) {

    for (int i = ndim-1; i >= 0; --i) {
        Real sum = 0.0;
        for (int j = i + 1; j < ndim; ++j)
            sum += LUMat[perm[i]*ndim + j] * outArr[j];
        outArr[i] = (inArr[i] - sum)/LUMat[perm[i]*ndim + i];
    }

    return;
}

//----------------------------------------------------------------------------------------
//! \fn  void InvertMatrix
//  \brief Compute the inverse of an nxn matrix as a 1D array. the input should also be a 1D representation of the matrix
KOKKOS_INLINE_FUNCTION
void InvertMatrixLU( const int ndim, const Real * inputMat, Real * outputMat, bool &fail ){

  // Because ndim is determined at runtime
  // Use 1D arrays and deal manually with column/row
  Real LUMat[6*6];
  Real x[6];
  Real y[6];
  Real e[6];
  int perm_arr[6];
  
  for (int ii = 0; ii<ndim*ndim; ++ii) {
    LUMat[ii] = inputMat[ii];
  }

  bool has_error = false;
  LUDecomposition(ndim, LUMat, perm_arr, has_error);
  if (has_error) {
    fail = true;
    return;
  }


  for (int i = 0; i < ndim; ++i) {
    for (int ii = 0; ii<ndim; ++ii)
      e[ii] = 0.0;
    e[i] = 1.0;

    FWDSubstitution(ndim, LUMat, perm_arr, e, x);
    BWDSubstitution(ndim, LUMat, perm_arr, x, y);

    for (int j = 0; j < ndim; ++j)
        outputMat[j*ndim + i] = y[j];
  }

  return;
}
#endif
