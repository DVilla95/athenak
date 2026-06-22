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
//! \brief
KOKKOS_INLINE_FUNCTION
static void GetUpperAdmMetric( const Real inputMat[][4], Real outputMat[][3] ){
	for (int i1 = 0; i1 < 3; ++i1 ){ 
		for (int i2 = 0; i2 < 3; ++i2 ){ 
		outputMat[i1][i2] = inputMat[i1+1][i2+1] - inputMat[0][i2+1]*inputMat[i1+1][0]/inputMat[0][0];
		}
	}
}

//----------------------------------------------------------------------------------------
//! \fn  void ComputeDeterminant3
//! \brief Compute the determinant of a 3x3 matrix
KOKKOS_INLINE_FUNCTION
static void ComputeDeterminant3( const Real inputMat[][3], Real &determinant ){

	determinant = inputMat[0][0]*inputMat[1][1]*inputMat[2][2] + inputMat[0][1]*inputMat[1][2]*inputMat[2][0]
		+ inputMat[1][0]*inputMat[2][1]*inputMat[0][2]
		- inputMat[2][0]*inputMat[1][1]*inputMat[0][2] - inputMat[1][0]*inputMat[0][1]*inputMat[2][2]
		- inputMat[2][1]*inputMat[1][2]*inputMat[1][1];

}

//----------------------------------------------------------------------------------------
//! \fn  void ComputeInverseMatrix3
//! \brief Compute the inverse of a 3x3 matrix
KOKKOS_INLINE_FUNCTION
static void ComputeInverseMatrix3( const Real inputMat[][3], Real outputMat[][3] ){

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
//! \brief Interpolate cell field to particle location.
//! Extension to full-trilinear interpolation done with LLM

KOKKOS_INLINE_FUNCTION
static void InterpolateFields( const Real * prtcl_x, const DvceFaceFld4D<Real> &b0_, const DvceEdgeFld4D<Real> &e0_,
                const DualArray1D<RegionSize> &mbsize, const RegionIndcs &indcs, const int m,
                Real * E, Real * B, bool &out_of_bounds ){

    // Cell index of the cell whose left face contains the particle
    int ip = (prtcl_x[0] - mbsize.d_view(m).x1min)/mbsize.d_view(m).dx1 + indcs.is;
    int jp = (prtcl_x[1] - mbsize.d_view(m).x2min)/mbsize.d_view(m).dx2 + indcs.js;
    int kp = (prtcl_x[2] - mbsize.d_view(m).x3min)/mbsize.d_view(m).dx3 + indcs.ks;

    if (ip < 1 || jp < 1 || kp < 1
        || ip > (indcs.ie + indcs.ng - 1) || jp > (indcs.je + indcs.ng - 1) || kp > (indcs.ke + indcs.ng - 1) ) {
        out_of_bounds = true;
        return;
    }

    Real &x1min = mbsize.d_view(m).x1min;
    Real &x2min = mbsize.d_view(m).x2min;
    Real &x3min = mbsize.d_view(m).x3min;
    Real &x1max = mbsize.d_view(m).x1max;
    Real &x2max = mbsize.d_view(m).x2max;
    Real &x3max = mbsize.d_view(m).x3max;
    Real Dx = (x1max - x1min)/indcs.nx1;
    Real Dy = (x2max - x2min)/indcs.nx2;
    Real Dz = (x3max - x3min)/indcs.nx3;

    // -----------------------------------------------------------------------
    // Compute weights for both face/edge and cell-center positions
    // -----------------------------------------------------------------------
    // Face/left-edge weights: node is at left face of cell ip,jp,kp
    Real x1L = LeftEdgeX(ip,  indcs.nx1, x1min, x1max);
    Real x2L = LeftEdgeX(jp,  indcs.nx2, x2min, x2max);
    Real x3L = LeftEdgeX(kp,  indcs.nx3, x3min, x3max);
    // wx1f in [0,1]: weight toward ip+1 face
    Real wx1f = (prtcl_x[0] - x1L)/Dx;
    Real wx2f = (prtcl_x[1] - x2L)/Dy;
    Real wx3f = (prtcl_x[2] - x3L)/Dz;

    // Cell-center weights: need to find which half of the cell we are in
    Real x1C = CellCenterX(ip, indcs.nx1, x1min, x1max);
    Real x2C = CellCenterX(jp, indcs.nx2, x2min, x2max);
    Real x3C = CellCenterX(kp, indcs.nx3, x3min, x3max);

    // For cell-centered staggering we interpolate between ip and ip+-1
    // Shift the base index if particle is in the lower half so we always
    // interpolate between ic and ic+1
    int ic = ip, jc = jp, kc = kp;
    Real wx1c = (prtcl_x[0] - x1C)/Dx;
    Real wx2c = (prtcl_x[1] - x2C)/Dy;
    Real wx3c = (prtcl_x[2] - x3C)/Dz;
    if (wx1c < 0.0) { ic -= 1; wx1c += 1.0; }
    if (wx2c < 0.0) { jc -= 1; wx2c += 1.0; }
    if (wx3c < 0.0) { kc -= 1; wx3c += 1.0; }

    // Complement weights
    Real wx1f_ = 1.0 - wx1f;
    Real wx2f_ = 1.0 - wx2f;
    Real wx3f_ = 1.0 - wx3f;
    Real wx1c_ = 1.0 - wx1c;
    Real wx2c_ = 1.0 - wx2c;
    Real wx3c_ = 1.0 - wx3c;

    // -----------------------------------------------------------------------
    // B1: staggered(x) x cell-center(y) x cell-center(z)
    //     => trilinear over (ip,ip+1) x (jc,jc+1) x (kc,kc+1)
    // -----------------------------------------------------------------------
    B[0] =
        wx1f_ * wx2c_ * wx3c_ * b0_.x1f(m, kc,   jc,   ip  ) +
        wx1f  * wx2c_ * wx3c_ * b0_.x1f(m, kc,   jc,   ip+1) +
        wx1f_ * wx2c  * wx3c_ * b0_.x1f(m, kc,   jc+1, ip  ) +
        wx1f  * wx2c  * wx3c_ * b0_.x1f(m, kc,   jc+1, ip+1) +
        wx1f_ * wx2c_ * wx3c  * b0_.x1f(m, kc+1, jc,   ip  ) +
        wx1f  * wx2c_ * wx3c  * b0_.x1f(m, kc+1, jc,   ip+1) +
        wx1f_ * wx2c  * wx3c  * b0_.x1f(m, kc+1, jc+1, ip  ) +
        wx1f  * wx2c  * wx3c  * b0_.x1f(m, kc+1, jc+1, ip+1);

    // -----------------------------------------------------------------------
    // B2: staggered(y) x cell-center(x) x cell-center(z)
    //     => trilinear over (ic,ic+1) x (jp,jp+1) x (kc,kc+1)
    // -----------------------------------------------------------------------
    B[1] =
        wx1c_ * wx2f_ * wx3c_ * b0_.x2f(m, kc,   jp,   ic  ) +
        wx1c  * wx2f_ * wx3c_ * b0_.x2f(m, kc,   jp,   ic+1) +
        wx1c_ * wx2f  * wx3c_ * b0_.x2f(m, kc,   jp+1, ic  ) +
        wx1c  * wx2f  * wx3c_ * b0_.x2f(m, kc,   jp+1, ic+1) +
        wx1c_ * wx2f_ * wx3c  * b0_.x2f(m, kc+1, jp,   ic  ) +
        wx1c  * wx2f_ * wx3c  * b0_.x2f(m, kc+1, jp,   ic+1) +
        wx1c_ * wx2f  * wx3c  * b0_.x2f(m, kc+1, jp+1, ic  ) +
        wx1c  * wx2f  * wx3c  * b0_.x2f(m, kc+1, jp+1, ic+1);

    // -----------------------------------------------------------------------
    // B3: staggered(z) x cell-center(x) x cell-center(y)
    //     => trilinear over (ic,ic+1) x (jc,jc+1) x (kp,kp+1)
    // -----------------------------------------------------------------------
    B[2] =
        wx1c_ * wx2c_ * wx3f_ * b0_.x3f(m, kp,   jc,   ic  ) +
        wx1c  * wx2c_ * wx3f_ * b0_.x3f(m, kp,   jc,   ic+1) +
        wx1c_ * wx2c  * wx3f_ * b0_.x3f(m, kp,   jc+1, ic  ) +
        wx1c  * wx2c  * wx3f_ * b0_.x3f(m, kp,   jc+1, ic+1) +
        wx1c_ * wx2c_ * wx3f  * b0_.x3f(m, kp+1, jc,   ic  ) +
        wx1c  * wx2c_ * wx3f  * b0_.x3f(m, kp+1, jc,   ic+1) +
        wx1c_ * wx2c  * wx3f  * b0_.x3f(m, kp+1, jc+1, ic  ) +
        wx1c  * wx2c  * wx3f  * b0_.x3f(m, kp+1, jc+1, ic+1);

    // -----------------------------------------------------------------------
    // E1: cell-center(x) x staggered(y) x staggered(z)
    //     => trilinear over (ic,ic+1) x (jp,jp+1) x (kp,kp+1)
    // -----------------------------------------------------------------------
    E[0] =
        wx1c_ * wx2f_ * wx3f_ * e0_.x1e(m, kp,   jp,   ic  ) +
        wx1c  * wx2f_ * wx3f_ * e0_.x1e(m, kp,   jp,   ic+1) +
        wx1c_ * wx2f  * wx3f_ * e0_.x1e(m, kp,   jp+1, ic  ) +
        wx1c  * wx2f  * wx3f_ * e0_.x1e(m, kp,   jp+1, ic+1) +
        wx1c_ * wx2f_ * wx3f  * e0_.x1e(m, kp+1, jp,   ic  ) +
        wx1c  * wx2f_ * wx3f  * e0_.x1e(m, kp+1, jp,   ic+1) +
        wx1c_ * wx2f  * wx3f  * e0_.x1e(m, kp+1, jp+1, ic  ) +
        wx1c  * wx2f  * wx3f  * e0_.x1e(m, kp+1, jp+1, ic+1);

    // -----------------------------------------------------------------------
    // E2: staggered(x) x cell-center(y) x staggered(z)
    //     => trilinear over (ip,ip+1) x (jc,jc+1) x (kp,kp+1)
    // -----------------------------------------------------------------------
    E[1] =
        wx1f_ * wx2c_ * wx3f_ * e0_.x2e(m, kp,   jc,   ip  ) +
        wx1f  * wx2c_ * wx3f_ * e0_.x2e(m, kp,   jc,   ip+1) +
        wx1f_ * wx2c  * wx3f_ * e0_.x2e(m, kp,   jc+1, ip  ) +
        wx1f  * wx2c  * wx3f_ * e0_.x2e(m, kp,   jc+1, ip+1) +
        wx1f_ * wx2c_ * wx3f  * e0_.x2e(m, kp+1, jc,   ip  ) +
        wx1f  * wx2c_ * wx3f  * e0_.x2e(m, kp+1, jc,   ip+1) +
        wx1f_ * wx2c  * wx3f  * e0_.x2e(m, kp+1, jc+1, ip  ) +
        wx1f  * wx2c  * wx3f  * e0_.x2e(m, kp+1, jc+1, ip+1);

    // -----------------------------------------------------------------------
    // E3: staggered(x) x staggered(y) x cell-center(z)
    //     => trilinear over (ip,ip+1) x (jp,jp+1) x (kc,kc+1)
    // -----------------------------------------------------------------------
    E[2] =
        wx1f_ * wx2f_ * wx3c_ * e0_.x3e(m, kc,   jp,   ip  ) +
        wx1f  * wx2f_ * wx3c_ * e0_.x3e(m, kc,   jp,   ip+1) +
        wx1f_ * wx2f  * wx3c_ * e0_.x3e(m, kc,   jp+1, ip  ) +
        wx1f  * wx2f  * wx3c_ * e0_.x3e(m, kc,   jp+1, ip+1) +
        wx1f_ * wx2f_ * wx3c  * e0_.x3e(m, kc+1, jp,   ip  ) +
        wx1f  * wx2f_ * wx3c  * e0_.x3e(m, kc+1, jp,   ip+1) +
        wx1f_ * wx2f  * wx3c  * e0_.x3e(m, kc+1, jp+1, ip  ) +
        wx1f  * wx2f  * wx3c  * e0_.x3e(m, kc+1, jp+1, ip+1);
}


//----------------------------------------------------------------------------------------
//! \fn  void LUDecomposition
//! \brief Compute matrix to use as Lower and Upper triangular decomposition in matrix inversion
KOKKOS_INLINE_FUNCTION
static void LUDecomposition( const int ndim, Real * LUMat, int * perm, bool &fail ){

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
//! \brief First step of the inversion
KOKKOS_INLINE_FUNCTION
static void FWDSubstitution(const int ndim, const Real * LUMat, const int * perm, const Real * idArr, Real * outArr) {

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
//! \brief Second step of the inversion
KOKKOS_INLINE_FUNCTION
static void BWDSubstitution(const int ndim, const Real * LUMat, const int * perm, const Real * inArr, Real * outArr) {

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
//! \brief Compute the inverse of an nxn matrix as a 1D array. the input should also be a 1D representation of the matrix
KOKKOS_INLINE_FUNCTION
static void InvertMatrixLU( const int ndim, const Real * inputMat, Real * outputMat, bool &fail ){

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

//----------------------------------------------------------------------------------------
// Function to initialize kinetic (i.e. 3 velocity components) particles.

KOKKOS_INLINE_FUNCTION
static void InjectKineticPrtcl( Real x1, Real x2, Real x3, Real * u, Real * b,
                       Real massive, Real q_o_m, Real this_en, Real max_en, Real min_en,
                       bool is_mnkwsk, Real bh_a, bool set_radius) {
    // u is contravariant in normal frame
    Real u_aux[3];
    Real gu[4][4], gl[4][4];
    ComputeMetricAndInverse( x1, x2, x3, is_mnkwsk, bh_a, gl, gu); 
    Real alpha = sqrt(-1.0/gu[0][0]);
    if (set_radius) {
      Real b_norm = gl[1][1]*SQR(b[0]) + gl[2][2]*SQR(b[1]) + gl[3][3]*SQR(b[2])
            + 2.0*gl[1][2]*b[0]*b[1] + 2.0*gl[1][3]*b[0]*b[2]
            + 2.0*gl[3][2]*b[2]*b[1];
      Real u0 = gl[1][1]*SQR(u[0]) + gl[2][2]*SQR(u[1]) + gl[3][3]*SQR(u[2])
            + 2.0*gl[1][2]*u[0]*u[1] + 2.0*gl[1][3]*u[0]*u[2]
            + 2.0*gl[3][2]*u[2]*u[1];
      u0 = sqrt(u0 + massive); // Lorentz factor in FIDO/normal frame
      // Lower indeces on velocity for scalar product with magnetic field
      u_aux[0] = gl[1][1]*u[0] + gl[1][2]*u[1] + gl[1][3]*u[2];
      u_aux[1] = gl[2][1]*u[0] + gl[2][2]*u[1] + gl[2][3]*u[2];
      u_aux[2] = gl[3][1]*u[0] + gl[3][2]*u[1] + gl[3][3]*u[2];
      for (int ii = 0; ii<3; ++ii)
        u_aux[ii] /= u0; // Get three-velocity 
      Real v_norm = b[0]*u_aux[0] + b[1]*u_aux[1] + b[2]*u_aux[2];
      for (int ii = 0; ii<3; ++ii) {
        u_aux[ii] = u[ii]/u0 - v_norm*b[ii]; // Get perpendicular velocity
        u_aux[ii] /= b_norm; // Normalize by magnetic field strength
      }
      
      v_norm = gl[1][1]*SQR(u_aux[0]) + gl[2][2]*SQR(u_aux[1]) + gl[3][3]*SQR(u_aux[2])
            + 2.0*gl[1][2]*u_aux[0]*u_aux[1] + 2.0*gl[1][3]*u_aux[0]*u_aux[2]
            + 2.0*gl[3][2]*u_aux[2]*u_aux[1];
      Real r_larmor = sqrt(v_norm)*u0/q_o_m/sqrt(b_norm); // Larmor radius computed with perpendicular 4-velocity
      Real fact = (r_larmor > this_en) ? 0.95 : 1.05;
      Real ggll = (r_larmor > this_en) ? 1.0 : -1.0;
      while (ggll*r_larmor > ggll*this_en) {
        u[0] *= fact;
        u[1] *= fact;
        u[2] *= fact;
        u0 = gl[1][1]*SQR(u[0]) + gl[2][2]*SQR(u[1]) + gl[3][3]*SQR(u[2])
              + 2.0*gl[1][2]*u[0]*u[1] + 2.0*gl[1][3]*u[0]*u[2]
              + 2.0*gl[3][2]*u[2]*u[1];
        u0 = sqrt(u0 + massive);
        u_aux[0] = gl[1][1]*u[0] + gl[1][2]*u[1] + gl[1][3]*u[2];
        u_aux[1] = gl[2][1]*u[0] + gl[2][2]*u[1] + gl[2][3]*u[2];
        u_aux[2] = gl[3][1]*u[0] + gl[3][2]*u[1] + gl[3][3]*u[2];
        for (int ii = 0; ii<3; ++ii)
          u_aux[ii] /= u0;
        v_norm = b[0]*u_aux[0] + b[1]*u_aux[1] + b[2]*u_aux[2];
        for (int ii = 0; ii<3; ++ii) {
          u_aux[ii] = u[ii]/u0 - v_norm*b[ii];
          u_aux[ii] /= b_norm;
        }
        
        v_norm = gl[1][1]*SQR(u_aux[0]) + gl[2][2]*SQR(u_aux[1]) + gl[3][3]*SQR(u_aux[2])
              + 2.0*gl[1][2]*u_aux[0]*u_aux[1] + 2.0*gl[1][3]*u_aux[0]*u_aux[2]
              + 2.0*gl[3][2]*u_aux[2]*u_aux[1];
        r_larmor = sqrt(v_norm)*u0/q_o_m/sqrt(b_norm); // Larmor radius computed with perpendicular 4-velocity
      }
      for (int ii = 0; ii<3; ++ii)
        u_aux[ii] = u[ii];
    } else {
      for (int ii = 0; ii<3; ++ii)
        u_aux[ii] = u[ii];
      Real u0 = gl[1][1]*SQR(u_aux[0]) + gl[2][2]*SQR(u_aux[1]) + gl[3][3]*SQR(u_aux[2])
            + 2.0*gl[1][2]*u_aux[0]*u_aux[1] + 2.0*gl[1][3]*u_aux[0]*u_aux[2]
            + 2.0*gl[3][2]*u_aux[2]*u_aux[1];
      u0 = sqrt(u0 + massive)/alpha; 
      Real fact = (u0 > this_en) ? 0.95 : 1.05;
      Real ggll = (u0 > this_en) ? 1.0 : -1.0;
      while (ggll*u0 > ggll*this_en) {
        u_aux[0] *= fact;
        u_aux[1] *= fact;
        u_aux[2] *= fact;
        u0 = gl[1][1]*SQR(u_aux[0]) + gl[2][2]*SQR(u_aux[1]) + gl[3][3]*SQR(u_aux[2])
              + 2.0*gl[1][2]*u_aux[0]*u_aux[1] + 2.0*gl[1][3]*u_aux[0]*u_aux[2]
              + 2.0*gl[3][2]*u_aux[2]*u_aux[1];
        u0 = sqrt(u0 + massive)/alpha; 
      }
    }
    // Velocity was contravariant in FIDO/normal frame
    // Lower indeces on velocity.
    // Covariant velocity in FIDO and coordinate frame match
    u[0] = gl[1][1]*u_aux[0] + gl[1][2]*u_aux[1] + gl[1][3]*u_aux[2];
    u[1] = gl[2][1]*u_aux[0] + gl[2][2]*u_aux[1] + gl[2][3]*u_aux[2];
    u[2] = gl[3][1]*u_aux[0] + gl[3][2]*u_aux[1] + gl[3][3]*u_aux[2];
}

#endif
