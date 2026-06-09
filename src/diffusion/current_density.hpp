#ifndef DIFFUSION_CURRENT_DENSITY_HPP_
#define DIFFUSION_CURRENT_DENSITY_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file current_density.hpp
//  \brief Inlined function to compute current density in 1-D pencils in i-direction

#include "athena.hpp"
#include "mesh/mesh.hpp"

//----------------------------------------------------------------------------------------
//! \fn CurrentDensity()
//  \brief Calculates the three components of the current density at cell edges
//  Each component of J is centered identically to the edge-electric-field
//               _____________
//               |\           \
//               | \           \
//               |  \___________\
//               |   |           |
//               \   |           |
//              J2*  *J3         |
//   x2 x3         \ |           |
//    \ |           \|_____*_____|
//     \|__x1             J1

KOKKOS_INLINE_FUNCTION
void CurrentDensity(TeamMember_t const &member, const int m, const int k, const int j,
     const int il, const int iu, const DvceFaceFld4D<Real> &b, const RegionSize &size,
     ScrArray1D<Real> &j1, ScrArray1D<Real> &j2, ScrArray1D<Real> &j3) {
  par_for_inner(member, il, iu, [&](const int i) {
    j1(i) = 0.0;
    j2(i) = -(b.x3f(m,k,j,i) - b.x3f(m,k,j,i-1))/size.dx1;
    j3(i) =  (b.x2f(m,k,j,i) - b.x2f(m,k,j,i-1))/size.dx1;
  });
  member.team_barrier();

  if (b.x1f.extent_int(2) > 1) {  // proxy for nx2gt1: 2D problems
    par_for_inner(member, il, iu, [&](const int i) {
      j1(i) += (b.x3f(m,k,j,i) - b.x3f(m,k,j-1,i))/size.dx2;
      j3(i) -= (b.x1f(m,k,j,i) - b.x1f(m,k,j-1,i))/size.dx2;
    });
    member.team_barrier();
  }

  if (b.x1f.extent_int(1) > 1) {  // proxy for nx3gt1: 3D problems
    par_for_inner(member, il, iu, [&](const int i) {
      j1(i) -= (b.x2f(m,k,j,i) - b.x2f(m,k-1,j,i))/size.dx3;
      j2(i) += (b.x1f(m,k,j,i) - b.x1f(m,k-1,j,i))/size.dx3;
    });
  }
  return;
}

KOKKOS_INLINE_FUNCTION
void CurlFC(const int m, const int k, const int j, const int i,
     const DvceFaceFld4D<Real> &b, const RegionSize &size,
     Real &j1, Real &j2, Real &j3,
     const Real thr_val, bool &thr_check ) {
  Real x3x1, x3x2, x2x3, x2x1, x1x3, x1x2;
  x3x1 = (b.x3f(m,k,j,i) - b.x3f(m,k,j,i-1))/size.dx1;
  x2x1 = (b.x2f(m,k,j,i) - b.x2f(m,k,j,i-1))/size.dx1;
  x3x2 = (b.x3f(m,k,j,i) - b.x3f(m,k,j-1,i))/size.dx2;
  x1x2 = (b.x1f(m,k,j,i) - b.x1f(m,k,j-1,i))/size.dx2;
  x2x3 = (b.x2f(m,k,j,i) - b.x2f(m,k-1,j,i))/size.dx3;
  x1x3 = (b.x1f(m,k,j,i) - b.x1f(m,k-1,j,i))/size.dx3;
  thr_check = thr_check 
    && ( fabs( 2.0*x1x2 / (b.x1f(m,k,j,i)+b.x1f(m,k,j-1,i)) ) < thr_val );
  thr_check = thr_check 
    && ( fabs( 2.0*x1x3 / (b.x1f(m,k,j,i)+b.x1f(m,k-1,j,i)) ) < thr_val );
  thr_check = thr_check 
    && ( fabs( 2.0*x2x1 / (b.x2f(m,k,j,i)+b.x2f(m,k,j,i-1)) ) < thr_val );
  thr_check = thr_check 
    && ( fabs( 2.0*x2x3 / (b.x2f(m,k,j,i)+b.x2f(m,k-1,j,i)) ) < thr_val );
  thr_check = thr_check 
    && ( fabs( 2.0*x3x1 / (b.x3f(m,k,j,i)+b.x3f(m,k,j,i-1)) ) < thr_val );
  thr_check = thr_check 
    && ( fabs( 2.0*x3x2 / (b.x3f(m,k,j,i)+b.x3f(m,k,j-1,i)) ) < thr_val );
  j1 = 0.0;
  j2 = -x3x1;
  j3 =  x2x1;

  j1 += x3x2;
  j3 -= x1x2;

  j1 -= x2x3;
  j2 += x1x3;
  return;
}

//Fancy sign function
KOKKOS_INLINE_FUNCTION
int SignOf(Real val) {
  return ( (val > 0.0) - (val < 0.0) );
}
#endif // DIFFUSION_CURRENT_DENSITY_HPP_
