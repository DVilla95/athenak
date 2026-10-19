//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file gr_pushers.cpp
//  \brief

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "particles.hpp"
#include "hamiltonian_gr.hpp"
#include "particles_helpers.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/cartesian_ks.hpp"

namespace particles {

//----------------------------------------------------------------------------------------
//! \fn  void Particles::HamiltonianGeodesicsIterations
//! \brief
//! Largely implemented following Bacchini et al. 2020 (https://doi.org/10.3847/1538-4365/abb604)
//! Computes the geodesic terms in the particle push using a conservative hamiltonian scheme
//! Robust and accurate but expensive
void Particles::HamiltonianGeodesicsIterations( const Real dt ){
  auto &pr = prtcl_rdata;
  auto &pi = prtcl_idata;
  const Real it_tol = iter_tolerance;
  const Real spin = pmy_pack->pcoord->coord_data.bh_spin;
  const int it_max = max_iter;
  const bool &multi_d = pmy_pack->pmesh->multi_d;
  const bool &three_d = pmy_pack->pmesh->three_d;
  auto gids = pmy_pack->gids;
  bool skip_em = false;
  const bool is_minkowski = pmy_pack->pcoord->coord_data.is_minkowski;

  const Real x_step = 1.0E-08;
  const Real v_step = 1.0E-08;
  Real avg_iter = 0.0;

  Kokkos::parallel_reduce("part_ham_geo",Kokkos::RangePolicy<>(DevExeSpace(),0,nprtcl_thispack),
    KOKKOS_LAMBDA(const int p, Real &aux_n_iter) {
  //par_for("part_fullgr",DevExeSpace(),0,(nprtcl_thispack-1),
  //KOKKOS_LAMBDA(const int p) {

    // Iterate per particle such that those that converge quicker don't go through as many iterations
    // Initialize iteration variables
    Real x_init[3] = {pr(IPX,p), pr(IPY,p), pr(IPZ,p)};
    Real v_init[3] = {pr(IPVX,p), pr(IPVY,p), pr(IPVZ,p)};
    Real x_eval[3] = {pr(IPX,p)+pr(IPVX,p)*dt/2.0, pr(IPY,p)+pr(IPVY,p)*dt/2.0, pr(IPZ,p)+pr(IPVZ,p)*dt/2.0};
    Real v_eval[3] = {pr(IPVX,p), pr(IPVY,p), pr(IPVZ,p)};
    Real x_grad[3], v_grad[3];
    //u is always contravariant. Iteration variables
    Real RHS_eval_v[3], RHS_eval_x[3]; 
    Real Jacob[3][3], inv_Jacob[3][3];
    Real RHS_grad_1[3], RHS_grad_2[3];
    Real res_v[3], res_x[3];
    int n_iter = 0;
    Real step_fac = 1.0;
    int m = pi(PGID,p) - gids;
    Real E[3], B[3];
    int i1, i2;

    Real glower[4][4], gupper[4][4], adm[3][3]; // Metric 
    Real gamma;
    ComputeMetricAndInverse(x_init[0],x_init[1],x_init[2], is_minkowski, spin, glower, gupper); 
    GetUpperAdmMetric( gupper, adm );
    gamma = adm[0][0]*SQR(v_init[0]) + adm[1][1]*SQR(v_init[1]) + adm[2][2]*SQR(v_init[2])
      + 2.0*adm[0][1]*v_init[0]*v_init[1] + 2.0*adm[0][2]*v_init[0]*v_init[2] + 2.0*adm[1][2]*v_init[1]*v_init[2];
    gamma = sqrt(1.0 + gamma)*sqrt(-gupper[0][0]);
    v_grad[0] = adm[0][0]*v_init[0] + adm[0][1]*v_init[1] + adm[0][2]*v_init[2];
    v_grad[1] = adm[1][0]*v_init[0] + adm[1][1]*v_init[1] + adm[1][2]*v_init[2];
    v_grad[2] = adm[2][0]*v_init[0] + adm[2][1]*v_init[1] + adm[2][2]*v_init[2];

    x_eval[0] = x_init[0] + dt/(12.0)*(v_grad[0]/gamma + gupper[0][1]/gupper[0][0]) ;
    x_eval[1] = x_init[1] + dt/(12.0)*(v_grad[1]/gamma + gupper[0][2]/gupper[0][0]) ;
    x_eval[2] = x_init[2] + dt/(12.0)*(v_grad[2]/gamma + gupper[0][3]/gupper[0][0]) ;

    // Start iterating
    // Using Newton method, thus computing the Jacobian at each iteration
    do{
      
    ++n_iter;
    if (n_iter > 5){ step_fac = 1E+3; }

    HamiltonEquation_Position(x_init, x_eval, v_init, v_eval, spin, RHS_eval_x);
    HamiltonEquation_Velocity(x_init, x_eval, v_init, v_eval, x_step, spin, it_tol, RHS_eval_v);

    // First Jacobian for position
    for (int dir = 0; dir<3; ++dir) {
      if (dir == 0) { i1 = 1; i2 = 2; }
      else if (dir == 1) { i1 = 0; i2 = 2; }
      else if (dir == 2) { i1 = 0; i2 = 1; }
      x_grad[dir] = x_eval[dir] + x_step;
      x_grad[i1] = x_eval[i1]; x_grad[i2] = x_eval[i2];
      HamiltonEquation_Position(x_init, x_grad, v_init, v_eval, spin, RHS_grad_1);
      x_grad[dir] = x_eval[dir] - x_step;
      HamiltonEquation_Position(x_init, x_grad, v_init, v_eval, spin, RHS_grad_2);
      for (int i=0; i<3; ++i) { Jacob[dir][i] = -(RHS_grad_1[i] - RHS_grad_2[i])*dt/(2.0*x_step); }
      Jacob[dir][dir] += 1.0; // Diagonal terms
    }
    ComputeInverseMatrix3( Jacob, inv_Jacob );

    // Store values for use with velocity Jacobian
    for (int i=0; i<3; ++i) {
      x_grad[i] = x_eval[i];
      res_x[i] = x_grad[i] - x_init[i] - RHS_eval_x[i]*dt;
    }

    for (int i=0; i<3; ++i){
      for (int j=0; j<3; ++j){ x_eval[i] -= inv_Jacob[i][j]*res_x[j]; }
    }

    // Then Jacobian for velocity
    for (int dir = 0; dir<3; ++dir) {
      if (dir == 0) { i1 = 1; i2 = 2; }
      else if (dir == 1) { i1 = 0; i2 = 2; }
      else if (dir == 2) { i1 = 0; i2 = 1; }
      v_grad[dir] = v_eval[dir] + v_step;
      v_grad[i1] = v_eval[i1]; v_grad[i2] = v_eval[i2];
      HamiltonEquation_Velocity(x_init, x_grad, v_init, v_grad, x_step, spin, it_tol, RHS_grad_1);
      v_grad[dir] = v_eval[dir] - v_step;
      HamiltonEquation_Velocity(x_init, x_grad, v_init, v_grad, x_step, spin, it_tol, RHS_grad_2);
      for (int i=0; i<3; ++i) { Jacob[dir][i] = -(RHS_grad_1[i] - RHS_grad_2[i])*dt/(2.0*v_step); }
      Jacob[dir][dir] += 1.0; // Diagonal terms
    }
    ComputeInverseMatrix3( Jacob, inv_Jacob );
    
    for (int i=0; i<3; ++i) {
      v_grad[i] = v_eval[i];
      res_v[i] = (v_grad[i] - v_init[i] - RHS_eval_v[i]*dt);
    }

    for (int i=0; i<3; ++i){
      for (int j=0; j<3; ++j){ v_eval[i] -= inv_Jacob[i][j]*res_v[j]; }
    }

    }while(
      n_iter < it_max
      && ( (SQR(res_x[0]) + SQR(res_x[1]) + SQR(res_x[2])) > it_tol
      || (SQR(res_v[0]) + SQR(res_v[1]) + SQR(res_v[2])) > it_tol )
         );

    // Done with iterations, update ``true'' values
    pr(IPVX,p) = v_eval[0];
    if (multi_d) { pr(IPVY,p) = v_eval[1]; }
    if (three_d) { pr(IPVZ,p) = v_eval[2]; }
    pr(IPX,p) = x_eval[0];
    if (multi_d) { pr(IPY,p) = x_eval[1]; }
    if (three_d) { pr(IPZ,p) = x_eval[2]; }
    aux_n_iter += n_iter;
  }, Kokkos::Sum<Real>(avg_iter));
  average_iteration_number += avg_iter / nprtcl_thispack;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void Particles::GRLorentzIterations
//! \brief
//! Largely implemented following Bacchini et al. 2019
//! Computes the geodesic terms couple to the electromagnetic terms in the particle push
//! using implicit mid-point rule iterations
//! Fast but not the most precise solution
void Particles::GRLorentzIterations( const Real dt ){
  auto &pr = prtcl_rdata;
  auto &pi = prtcl_idata;
  const Real it_tol = iter_tolerance;
  const Real spin = pmy_pack->pcoord->coord_data.bh_spin;
  const int it_max = max_iter;
  const bool &multi_d = pmy_pack->pmesh->multi_d;
  const bool &three_d = pmy_pack->pmesh->three_d;
  auto gids = pmy_pack->gids;
  auto gide = pmy_pack->gide;
  const Real &q_over_m = charge_over_mass;
  auto &b0_ = pmy_pack->pmhd->b0;
  auto &e0_ = pmy_pack->pmhd->efld;
  const bool is_minkowski = pmy_pack->pcoord->coord_data.is_minkowski;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &mbsize = pmy_pack->pmb->mb_size;

  const Real base_v_step = 1.0e-6;
  const Real base_x_step = 1.0e-6;
  const Real damp_min = 1.0e-6;
  int avg_iter = 0;
  int tot_max_iter = 0;
  int tot_n_fails = 0;
  int ndim = 6;

  Kokkos::parallel_reduce("part_grlorentz",Kokkos::RangePolicy<>(DevExeSpace(),0,nprtcl_thispack),
    KOKKOS_LAMBDA(const int p, int &aux_n_iter, int &max_n_iter, int &aux_n_fails) {

    // Iterate per particle such that those that converge quicker don't go through as many iterations
    // Initialize iteration variables
    const Real x_init[3] = {pr(IPX,p), pr(IPY,p), pr(IPZ,p)};
    const Real v_init[3] = {pr(IPVX,p), pr(IPVY,p), pr(IPVZ,p)};
    Real x_grad[3] = {pr(IPX,p), pr(IPY,p), pr(IPZ,p)};
    Real v_grad[3] = {pr(IPVX,p), pr(IPVY,p), pr(IPVZ,p)};
    Real x_eval[3] = {pr(IPX,p)+pr(IPVX,p)*dt/2.0, pr(IPY,p)+pr(IPVY,p)*dt/2.0, pr(IPZ,p)+pr(IPVZ,p)*dt/2.0};
    Real v_eval[3] = {pr(IPVX,p), pr(IPVY,p), pr(IPVZ,p)};
    Real x_mid[3], v_mid[3];
    Real RHS_eval_v[3], RHS_eval_x[3]; 
    Real Jacob1D[6*6];
    Real invJacob1D[6*6];
    Real RHS_grad_x1[3], RHS_grad_x2[3], RHS_grad_v1[3], RHS_grad_v2[3];
    Real res[6];
    int n_iter = 1;
    int m = pi(PGID,p) - gids;
    Real E[3] = {0.0};
    Real B[3] = {0.0};
    Real resnorm, resold;
    int i1, i2;
    bool out_of_bounds = false; // Particle could leave MB during iterations (most likely Newton method not converging)
    bool invert_mat_fail = false; // Make sure Jacobians don't become singular
    Real x_save[3], v_save[3];
    const Real scales[6] = { 
      std::fmax(1.0, std::abs(x_init[0])), std::fmax(1.0, std::abs(x_init[1])), 
      std::fmax(1.0, std::abs(x_init[2])), std::fmax(1.0, std::abs(v_init[0])), 
      std::fmax(1.0, std::abs(v_init[1])), std::fmax(1.0, std::abs(v_init[2]))
    };
    Real scaled_update_x, scaled_update_v;

    GRRHSPosition(x_init, v_init, is_minkowski, spin, RHS_eval_x);
    GRRHSVelocity(x_init, v_init, is_minkowski, spin, RHS_eval_v);
    InterpolateFields(x_init, b0_, e0_, mbsize, indcs, m, E, B, out_of_bounds);
    GRLorentz_Terms(x_init, v_init, E, B, is_minkowski, spin, q_over_m, RHS_eval_v);
    for (int i = 0; i<3; ++i) {
      x_eval[i] = x_init[i] + 0.5*dt*RHS_eval_x[i];
      v_eval[i] = v_init[i] + 0.5*dt*RHS_eval_v[i];
    }

    GRRHSPosition(x_eval, v_eval, is_minkowski, spin, RHS_eval_x);
    GRRHSVelocity(x_eval, v_eval, is_minkowski, spin, RHS_eval_v);
    InterpolateFields(x_eval, b0_, e0_, mbsize, indcs, m, E, B, out_of_bounds);
    GRLorentz_Terms(x_eval, v_eval, E, B, is_minkowski, spin, q_over_m, RHS_eval_v);
    for (int i = 0; i<3; ++i) {
      x_eval[i] = x_init[i] + dt*RHS_eval_x[i];
      v_eval[i] = v_init[i] + dt*RHS_eval_v[i];
    }

    for (int i=0; i<3; ++i) {
      res[i] = (x_eval[i] - x_init[i] - RHS_eval_x[i]*dt);
      res[i+3] = (v_eval[i] - v_init[i] - RHS_eval_v[i]*dt);
    }
    for (int i=0; i<6; ++i) {res[i] /= scales[i];}
    resnorm = 0.0;
    for (int i=0; i<ndim; ++i)
      resnorm += SQR(res[i]);
    resold = resnorm;
    // Store initial residuals, so even if iteration hasn't converged to desired 
    // accuracy but has improved wrt initial guess you can keep the result rather than 
    // fall-back to explicit mid-point update
    const Real resinit = resnorm;

    Real x_step, v_step;
    Real x_fac, v_fac;
    x_fac = 1.0;
    v_fac = 1.0;

    // Start iterating
    // Using Newton method, thus computing the Jacobian at each iteration
    for (int i = 0; i<3; ++i) {
      x_mid[i] = 0.5*(x_eval[i] + x_init[i]);
      v_mid[i] = 0.5*(v_eval[i] + v_init[i]);
    }
    do{
    out_of_bounds = false; // Reset check variables
    invert_mat_fail = false;
    ++n_iter;
      
    // Position
    for (int dir = 0; dir<3; ++dir) {
      Real dirdx;
      const int full_idx = dir;
      const int row_idx = full_idx*ndim;
      if      (dir == 0) { i1 = 1; i2 = 2; dirdx = mbsize.d_view(m).dx1; }
      else if (dir == 1) { i1 = 0; i2 = 2; dirdx = mbsize.d_view(m).dx2; }
      else if (dir == 2) { i1 = 0; i2 = 1; dirdx = mbsize.d_view(m).dx3; }
      x_step = std::fmax(base_x_step, std::abs(x_init[dir]*x_fac*base_x_step));
      x_step = std::fmin(x_step, dirdx);

      x_grad[dir] = x_eval[dir] + x_step;
      x_grad[i1] = x_eval[i1]; x_grad[i2] = x_eval[i2];
      GRRHSPosition(x_grad, v_mid, is_minkowski, spin, RHS_grad_x1);
      GRRHSVelocity(x_grad, v_mid, is_minkowski, spin, RHS_grad_v1);
      InterpolateFields(x_grad, b0_, e0_, mbsize, indcs, m, E, B, out_of_bounds);
      if (out_of_bounds) {break;}
      GRLorentz_Terms(x_grad, v_mid, E, B, is_minkowski, spin, q_over_m, RHS_grad_v1);
      x_grad[dir] = x_eval[dir] - x_step;
      GRRHSPosition(x_grad, v_mid, is_minkowski, spin, RHS_grad_x2);
      GRRHSVelocity(x_grad, v_mid, is_minkowski, spin, RHS_grad_v2);
      InterpolateFields(x_grad, b0_, e0_, mbsize, indcs, m, E, B, out_of_bounds);
      if (out_of_bounds) {break;}
      GRLorentz_Terms(x_grad, v_mid, E, B, is_minkowski, spin, q_over_m, RHS_grad_v2);
      for (int i=0; i<3; ++i) { // Here Jacobian is for full system, position + velocity
        Jacob1D[row_idx + i]      = -(RHS_grad_x1[i] - RHS_grad_x2[i])*dt/(2.0*x_step);
        Jacob1D[row_idx + i + 3]  = -(RHS_grad_v1[i] - RHS_grad_v2[i])*dt/(2.0*x_step);
      }
      for (int i=0; i<6; ++i) {Jacob1D[row_idx + i] /= scales[i];}
      Jacob1D[row_idx + full_idx] += 1.0 + base_x_step;  // Diagonal terms with Tikhonov
    }

    if (!out_of_bounds) {
      InterpolateFields(x_mid, b0_, e0_, mbsize, indcs, m, E, B, out_of_bounds);
    }

    if (!out_of_bounds) {
      // Velocity
      for (int dir = 0; dir<3; ++dir) {
        const int full_idx = dir + 3;
        const int row_idx = full_idx*ndim;
        if      (dir == 0) { i1 = 1; i2 = 2; }
        else if (dir == 1) { i1 = 0; i2 = 2; }
        else if (dir == 2) { i1 = 0; i2 = 1; }
        v_step = std::fmax(base_v_step, std::abs(v_init[dir]*v_fac*base_v_step));
        // v_step = std::fmin(v_step, std::abs(v_init[dir]*0.1));

        v_grad[dir] = v_eval[dir] + v_step;
        v_grad[i1] = v_eval[i1]; v_grad[i2] = v_eval[i2];
        GRRHSPosition(x_mid, v_grad, is_minkowski, spin, RHS_grad_x1);
        GRRHSVelocity(x_mid, v_grad, is_minkowski, spin, RHS_grad_v1);
        GRLorentz_Terms(x_mid, v_grad, E, B, is_minkowski, spin, q_over_m, RHS_grad_v1);
        v_grad[dir] = v_eval[dir] - v_step;
        GRRHSPosition(x_mid, v_grad, is_minkowski, spin, RHS_grad_x2);
        GRRHSVelocity(x_mid, v_grad, is_minkowski, spin, RHS_grad_v2);
        GRLorentz_Terms(x_mid, v_grad, E, B, is_minkowski, spin, q_over_m, RHS_grad_v2);
        for (int i=0; i<3; ++i) { // Here Jacobian is for full system, position + velocity
          Jacob1D[row_idx + i]      = -(RHS_grad_x1[i] - RHS_grad_x2[i])*dt/(2.0*v_step);
          Jacob1D[row_idx + i + 3]  = -(RHS_grad_v1[i] - RHS_grad_v2[i])*dt/(2.0*v_step);
        }
        for (int i=0; i<6; ++i) {Jacob1D[row_idx + i] /= scales[i];}
        Jacob1D[row_idx + full_idx] += 1.0 + base_v_step; // Diagonal terms with Tikhonov
      }
    
      InvertMatrixLU( ndim, Jacob1D, invJacob1D, invert_mat_fail );
      if (invert_mat_fail) {
        x_fac *= 1.5 ;
        v_fac *= 1.5;
        continue;
      }

      for (int i = 0; i<3; ++i) {
        x_mid[i] = 0.5*(x_eval[i] + x_init[i]);
        v_mid[i] = 0.5*(v_eval[i] + v_init[i]);
      }
      GRRHSPosition(x_mid, v_mid, is_minkowski, spin, RHS_eval_x);
      GRRHSVelocity(x_mid, v_mid, is_minkowski, spin, RHS_eval_v);
      // InterpolateFields( pi(PTAG,p), x_eval, b0_, e0_, mbsize, indcs, m, E, B );
      GRLorentz_Terms(x_mid, v_mid, E, B, is_minkowski, spin, q_over_m, RHS_eval_v);
    }

    if (out_of_bounds) {
      // Reset
      GRRHSPosition(x_init, v_init, is_minkowski, spin, RHS_eval_x);
      GRRHSVelocity(x_init, v_init, is_minkowski, spin, RHS_eval_v);
      InterpolateFields(x_init, b0_, e0_, mbsize, indcs, m, E, B, out_of_bounds);
      GRLorentz_Terms(x_init, v_init, E, B, is_minkowski, spin, q_over_m, RHS_eval_v);
      x_fac *= 0.5;
      for (int i=0; i<3; ++i) {
        x_eval[i] = x_init[i] + RHS_eval_x[i]*dt;
        v_eval[i] = v_init[i] + RHS_eval_v[i]*dt;
      }
      continue;
    }

    for (int i=0; i<3; ++i) {
      res[i]   = (x_eval[i] - x_init[i] - RHS_eval_x[i]*dt);
      res[i+3] = (v_eval[i] - v_init[i] - RHS_eval_v[i]*dt);
    }
    for (int i=0; i<ndim; ++i) {res[i] /= scales[i];}
    resnorm = 0.0;
    for (int i = 0; i<ndim; ++i) {resnorm += SQR(res[i]);}

    if (resnorm < resold) {
      // Regular iteration
      for (int i=0; i<3; ++i){
        scaled_update_x = 0.0;
        scaled_update_v = 0.0;
        for (int j=0; j<6; ++j){
          scaled_update_x -= invJacob1D[j*ndim + i]*res[j];
          scaled_update_v -= invJacob1D[j*ndim + (i+3)]*res[j];
        }
        x_eval[i] += scaled_update_x;
        v_eval[i] += scaled_update_v;
      }
      resold = resnorm;
    } else {
      // If something went wrong, do backtracking
      Real damp_f = 1.0;
      bool resume_iter = false;
      for (int i=0; i<3; ++i) {
        x_save[i] = x_eval[i];
        v_save[i] = v_eval[i];
      }
      while (damp_f > damp_min && !resume_iter) {
        for (int i=0; i<3; ++i){
          x_eval[i] = x_save[i];
          v_eval[i] = v_save[i];
          scaled_update_x = 0.0;
          scaled_update_v = 0.0;
          for (int j=0; j<6; ++j){
            scaled_update_x -= damp_f*invJacob1D[j*ndim + i]*res[j];
            scaled_update_v -= damp_f*invJacob1D[j*ndim + (i+3)]*res[j];
          }
          x_eval[i] += scaled_update_x;
          v_eval[i] += scaled_update_v;
        }
        for (int i = 0; i<3; ++i) {
          x_mid[i] = 0.5*(x_eval[i] + x_init[i]);
          v_mid[i] = 0.5*(v_eval[i] + v_init[i]);
        }
        GRRHSPosition(x_mid, v_mid, is_minkowski, spin, RHS_eval_x);
        GRRHSVelocity(x_mid, v_mid, is_minkowski, spin, RHS_eval_v);
        InterpolateFields( x_mid, b0_, e0_, mbsize, indcs, m, E, B, out_of_bounds );
        if (out_of_bounds) {
          damp_f *= 0.5;
          continue;
        }
        GRLorentz_Terms(x_mid, v_mid, E, B, is_minkowski, spin, q_over_m, RHS_eval_v);
        for (int i=0; i<3; ++i) {
          res[i]   = (x_eval[i] - x_init[i] - RHS_eval_x[i]*dt);
          res[i+3] = (v_eval[i] - v_init[i] - RHS_eval_v[i]*dt);
        }
        for (int i=0; i<6; ++i) {res[i] /= scales[i];}
        resnorm = 0.0;
        for (int i=0; i<6; ++i) {resnorm += SQR(res[i]);}
        if (resnorm < resold) {
          resume_iter = true;
          resold = resnorm;
        } 
        else {damp_f *= 0.5;}
      }

      if (!resume_iter) {
        // Reset with "random" values
        x_fac *= 1.5;
        v_fac *= 1.5;
        for (int i=0; i<3; ++i) {
          x_eval[i] = x_init[i] + RHS_eval_x[i]*dt;
          v_eval[i] = v_init[i] + RHS_eval_v[i]*dt;
        }
      }
    }

    }while( (n_iter < it_max) && (resnorm > it_tol) ); // while

    // Done with iterations, update ``true'' values
    if (n_iter == it_max) {
      aux_n_fails++;
      // Check whether iterations have converged but not to desired accuracy
      // Or whether iterative process actually went badly
      // TODO: Track partial fails?
      if (resnorm > resinit) {
        // Fallback to mid-point explicit update
        GRRHSPosition(x_init, v_init, is_minkowski, spin, RHS_eval_x);
        GRRHSVelocity(x_init, v_init, is_minkowski, spin, RHS_eval_v);
        InterpolateFields(x_init, b0_, e0_, mbsize, indcs, m, E, B, out_of_bounds);
        GRLorentz_Terms(x_init, v_init, E, B, is_minkowski, spin, q_over_m, RHS_eval_v);
        for (int i = 0; i<3; ++i) {
          x_mid[i] = x_init[i] + 0.5*dt*RHS_eval_x[i];
          v_mid[i] = v_init[i] + 0.5*dt*RHS_eval_v[i];
        }
        GRRHSPosition(x_mid, v_mid, is_minkowski, spin, RHS_eval_x);
        GRRHSVelocity(x_mid, v_mid, is_minkowski, spin, RHS_eval_v);
        InterpolateFields(x_mid, b0_, e0_, mbsize, indcs, m, E, B, out_of_bounds);
        GRLorentz_Terms(x_mid, v_mid, E, B, is_minkowski, spin, q_over_m, RHS_eval_v);
        for (int i = 0; i<3; ++i) {
          x_eval[i] = x_init[i] + dt*RHS_eval_x[i];
          v_eval[i] = v_init[i] + dt*RHS_eval_v[i];
        }
      }
    }
    pr(IPVX,p) = v_eval[0];
    if (multi_d) { pr(IPVY,p) = v_eval[1]; }
    if (three_d) { pr(IPVZ,p) = v_eval[2]; }
    pr(IPX,p) = x_eval[0];
    if (multi_d) { pr(IPY,p) = x_eval[1]; }
    if (three_d) { pr(IPZ,p) = x_eval[2]; }
    aux_n_iter += n_iter;
    max_n_iter = std::fmax(max_n_iter, n_iter);
  }, Kokkos::Sum<int>(avg_iter), Kokkos::Max<int>(tot_max_iter), Kokkos::Sum<int>(tot_n_fails));
  if (nprtcl_thispack > 0) { average_iteration_number = static_cast<Real>(avg_iter)/nprtcl_thispack; }
  else { 
    average_iteration_number = 0.0; 
    tot_max_iter = 0;
    tot_n_fails = 0;
  }
  fail_num = tot_n_fails;
  max_iteration_number = tot_max_iter;
  return;
}

} // namespace particles
