//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file boris_gr.cpp
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
//! \fn  void Particles::BorisStepGR
//  \brief
//Provide dt and only_v as input parameter in order to be able to use this function
//also for half-steps in the full_gr pusher
//Largely implemented following Ripperda et al. 2018 (https://doi.org/10.3847/1538-4365/aab114)
void Particles::BorisStepGR( const Real dt, const bool only_v ){
  
  auto &npart = nprtcl_thispack;
  auto &pi = prtcl_idata;
  auto &pr = prtcl_rdata;
  auto &b0_ = pmy_pack->pmhd->b0;
  auto &e0_ = pmy_pack->pmhd->efld;
  const bool is_minkowski = pmy_pack->pcoord->coord_data.is_minkowski;
  const Real spin = pmy_pack->pcoord->coord_data.bh_spin;
  const bool &multi_d = pmy_pack->pmesh->multi_d;
  const bool &three_d = pmy_pack->pmesh->three_d;
  const Real &q_over_m = charge_over_mass;
  auto gids = pmy_pack->gids;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &mbsize = pmy_pack->pmb->mb_size;

  // First half-step in space
  par_for("part_boris",DevExeSpace(),0,npart-1,
  KOKKOS_LAMBDA(const int p) {
      
    // Co-variant 4-velocity in the coordinate frame
    // To have compatibility with GR the velocity stored should be the covariant one
    Real u_cov[3] = {pr(IPVX,p), pr(IPVY,p), pr(IPVZ,p)};
    Real x[3] =  {pr(IPX,p), pr(IPY,p), pr(IPZ,p)}; // Position.
    Real g_Lor;
    bool out_of_bounds = false;
    // Get metric components at starting location x1,x2,x3
    Real glower[4][4], gupper[4][4], ADM_upper[3][3]; // Metric 
                   // (remember: sqrt(-1/gupper[0][0]) = alpha)
    ComputeMetricAndInverse(x[0],x[1],x[2], is_minkowski, spin, glower, gupper); 
    // Compute 3x3 ADM spatial metric from metric 
    GetUpperAdmMetric( gupper, ADM_upper );
    //Lorentz factor in Normal frame
    g_Lor = ADM_upper[0][0]*SQR(u_cov[0]) + ADM_upper[1][1]*SQR(u_cov[1]) + ADM_upper[2][2]*SQR(u_cov[2])
      + 2.0*ADM_upper[0][1]*u_cov[0]*u_cov[1] + 2.0*ADM_upper[0][2]*u_cov[0]*u_cov[2] + 2.0*ADM_upper[1][2]*u_cov[1]*u_cov[2];
    // In principle the 1.0 should be replaced by a 0 if
    // the particle is massless, but I don't know of 
    // any massless particle that can interact with an 
    // electromagnetic field (unless one goes into quantum mechanics)
    // Convert to Lorentz factor in coordinate frame i.e u^0
    g_Lor = sqrt(1.0 + g_Lor)*sqrt(-gupper[0][0]);

    Real aux_vec[3] = {0.0};
    //Raise indeces to contravariant for position push
    for (int i1 = 0; i1 < 3; ++i1 ){ 
      for (int i2 = 0; i2 < 3; ++i2 ){ 
      aux_vec[i1] += ADM_upper[i1][i2]*u_cov[i2];
      }
    }
    x[0] = pr(IPX,p) + dt/(2.0)*(aux_vec[0]/g_Lor + gupper[0][1]/gupper[0][0]) ;
    if (multi_d) { x[1] = pr(IPY,p) + dt/(2.0)*(aux_vec[1]/g_Lor + gupper[0][2]/gupper[0][0]) ; }
    if (three_d) { x[2] = pr(IPZ,p) + dt/(2.0)*(aux_vec[2]/g_Lor + gupper[0][3]/gupper[0][0]) ; }

    int m = pi(PGID,p) - gids;
    Real uE[3]; //Evolution of the velocity due to the electric field (first half).
    Real uB[3]; //Evolution of the velocity due to the magnetic field.
    Real E[3], B[3];
    InterpolateFields( x, b0_, e0_, mbsize, indcs, m, E, B, out_of_bounds );

    // Get metric components at new location x
    ComputeMetricAndInverse(x[0],x[1],x[2], is_minkowski, spin, glower, gupper); 
    GetUpperAdmMetric( gupper, ADM_upper );
    // Determinant of metric needed for vector products
    Real adm_det; 
    ComputeDeterminant3( ADM_upper, adm_det );
    // Determinant needed is that of covariant metric
    adm_det = 1.0/adm_det;
    adm_det = sqrt(adm_det);

    // Electric field is stored in coordinate frame, need to combine with B to operate on velocity
    // Vector product of two controvariant vectors results in covariant vector
    Real E_beta[3] = {
      - gupper[0][2]/gupper[0][0]*B[2] + B[1]*gupper[0][3]/gupper[0][0],
      - gupper[0][3]/gupper[0][0]*B[0] + B[2]*gupper[0][1]/gupper[0][0],
      - gupper[0][1]/gupper[0][0]*B[1] + B[0]*gupper[0][2]/gupper[0][0]
    };
    for (int i = 0; i < 3; ++i ){ E_beta[i] *= adm_det; }
    for (int i = 0; i < 3; ++i ){ E[i] -= E_beta[i]; } //This is now alpha x D_i
    //Lower indeces of E to covariant for velocity push
    for (int i1 = 0; i1 < 3; ++i1 ){ 
      aux_vec[i1] = 0.0;
      for (int i2 = 0; i2 < 3; ++i2 ){ 
      aux_vec[i1] += glower[i1+1][i2+1]*E[i2];
      }
    }

    // Push 4-velocity with D_i field (first-half)
    uE[0] = u_cov[0] + dt*q_over_m/(2.0)*aux_vec[0];
    if (multi_d) { uE[1] = u_cov[1] + dt*q_over_m/(2.0)*aux_vec[1]; }
    if (three_d) { uE[2] = u_cov[2] + dt*q_over_m/(2.0)*aux_vec[2]; }

    //Intermediate Lorentz gamma factor
    g_Lor = ADM_upper[0][0]*SQR(uE[0]) + ADM_upper[1][1]*SQR(uE[1]) + ADM_upper[2][2]*SQR(uE[2])
      + 2.0*ADM_upper[0][1]*uE[0]*uE[1] + 2.0*ADM_upper[0][2]*uE[0]*uE[2] + 2.0*ADM_upper[1][2]*uE[1]*uE[2];
    g_Lor = sqrt(1.0 + g_Lor)*sqrt(-gupper[0][0]);

    // Rotation of velocity due to magnetic field done in 2 steps
    // i.e. Boris algorithm
    Real mod_t_sqr = 0.0;
    Real t[3];
    for (int i1 = 0; i1 < 3; ++i1 ){ t[i1] = B[i1]*q_over_m/(2.0*g_Lor)*dt; }
    // for (int i1 = 0; i1 < 3; ++i1 ){ 
    //   for (int i2 = 0; i2 < 3; ++i2 ){ 
    //   mod_t_sqr += glower[i1+1][i2+1]*t[i1]*t[i2];
    //   }
    // }
    mod_t_sqr = SQR(t[0]) + SQR(t[1]) + SQR(t[2]);

    Real uE_con[3] = {0.0}; //Raise indeces of uE for vector product
    for (int i1 = 0; i1 < 3; ++i1 ){ 
      for (int i2 = 0; i2 < 3; ++i2 ){ 
      uE_con[i1] += ADM_upper[i1][i2]*uE[i2];
      }
    }

    // Save the vector product of u and t 
    aux_vec[0] = uE_con[1]*t[2] - uE_con[2]*t[1];
    aux_vec[1] = uE_con[2]*t[0] - uE_con[0]*t[2];
    aux_vec[2] = uE_con[0]*t[1] - uE_con[1]*t[0];
    // Used a vector product, correct for volume
    for (int i = 0; i < 3; ++i ){ aux_vec[i] *= adm_det; }
    // Re-use arrays
    B[0] = (uE_con[1] + aux_vec[1])*t[2] - (uE_con[2] + aux_vec[2])*t[1];
    B[1] = (uE_con[2] + aux_vec[2])*t[0] - (uE_con[0] + aux_vec[0])*t[2];
    B[2] = (uE_con[0] + aux_vec[0])*t[1] - (uE_con[1] + aux_vec[1])*t[0];
    for (int i = 0; i < 3; ++i ){ B[i] *= adm_det; }

    aux_vec[0] = glower[1][1]*B[0] + glower[1][2]*B[1] + glower[1][3]*B[2];
    aux_vec[1] = glower[2][1]*B[0] + glower[2][2]*B[1] + glower[2][3]*B[2];
    aux_vec[2] = glower[3][1]*B[0] + glower[3][2]*B[1] + glower[3][3]*B[2];

    // Finalize rotation
    uB[0] = uE[0] + 2.0/(1.0+mod_t_sqr)*( aux_vec[0] );
    if (multi_d) { uB[1] = uE[1] + 2.0/(1.0+mod_t_sqr)*( aux_vec[1] ); }
    if (three_d) { uB[2] = uE[2] + 2.0/(1.0+mod_t_sqr)*( aux_vec[2] ); }

    //Second half-step with shifted electric field
    uE[0] = uB[0] + dt*q_over_m/(2.0)*E[0];
    if (multi_d) { uE[1] = uB[1] + dt*q_over_m/(2.0)*E[1]; }
    if (three_d) { uE[2] = uB[2] + dt*q_over_m/(2.0)*E[2]; }

    // Finally update velocity of particle
    pr(IPVX,p) = uE[0];
    pr(IPVY,p) = uE[1];
    pr(IPVZ,p) = uE[2];

    if (!only_v){
    //Final Lorentz gamma factor
    //Position is unchanged
    g_Lor = ADM_upper[0][0]*SQR(uE[0]) + ADM_upper[1][1]*SQR(uE[1]) + ADM_upper[2][2]*SQR(uE[2])
      + 2.0*ADM_upper[0][1]*uE[0]*uE[1] + 2.0*ADM_upper[0][2]*uE[0]*uE[2] + 2.0*ADM_upper[1][2]*uE[1]*uE[2];
    g_Lor = sqrt(1.0 + g_Lor)*sqrt(-gupper[0][0]);
    // Raise indeces of uE to update position
    for (int i1 = 0; i1 < 3; ++i1 ){ 
      aux_vec[i1] = 0.0;
      for (int i2 = 0; i2 < 3; ++i2 ){ 
      aux_vec[i1] += ADM_upper[i1][i2]*uE[i2];
      }
    }
    pr(IPX,p) = x[0] + dt/(2.0)*(aux_vec[0]/g_Lor + gupper[0][1]/gupper[0][0]) ;
    if (multi_d) { pr(IPY,p) = x[1] + dt/(2.0)*(aux_vec[1]/g_Lor + gupper[0][2]/gupper[0][0]) ; }
    if (three_d) { pr(IPZ,p) = x[2] + dt/(2.0)*(aux_vec[2]/g_Lor + gupper[0][3]/gupper[0][0]) ; }
    }
  });
  return;
}

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
    Real Jacob[6][6];
    Real invJacob1D[6*6];
    Real Jacob1D[6*6];
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
      std::abs(x_init[0]), std::abs(x_init[1]), std::abs(x_init[2]),
      std::abs(v_init[0]), std::abs(v_init[1]), std::abs(v_init[2])
    };
    Real scaled_update_x[3] = {0.0};
    Real scaled_update_v[3] = {0.0};

    GRRHSPosition(x_init, v_init, is_minkowski, spin, RHS_eval_x);
    x_eval[0] = x_init[0] + dt*(RHS_eval_x[0]) ;
    x_eval[1] = x_init[1] + dt*(RHS_eval_x[1]) ;
    x_eval[2] = x_init[2] + dt*(RHS_eval_x[2]) ;
    GRRHSVelocity(x_init, v_init, is_minkowski, spin, RHS_eval_v);
    InterpolateFields( x_init, b0_, e0_, mbsize, indcs, m, E, B, out_of_bounds );
    GRLorentz_Terms(x_init, v_init, E, B, is_minkowski, spin, q_over_m, RHS_eval_v);
    v_eval[0] = v_init[0] + dt*(RHS_eval_v[0]) ;
    v_eval[1] = v_init[1] + dt*(RHS_eval_v[1]) ;
    v_eval[2] = v_init[2] + dt*(RHS_eval_v[2]) ;
    for (int i = 0; i<3; ++i) {
      x_mid[i] = 0.5*(x_eval[i] + x_init[i]);
      v_mid[i] = 0.5*(v_eval[i] + v_init[i]);
    }

    GRRHSPosition(x_mid, v_mid, is_minkowski, spin, RHS_eval_x);
    GRRHSVelocity(x_mid, v_mid, is_minkowski, spin, RHS_eval_v);
    InterpolateFields(x_mid, b0_, e0_, mbsize, indcs, m, E, B, out_of_bounds);
    GRLorentz_Terms(x_mid, v_mid, E, B, is_minkowski, spin, q_over_m, RHS_eval_v);

    for (int i=0; i<3; ++i) {
      res[i] = (x_eval[i] - x_init[i] - RHS_eval_x[i]*dt);
      res[i+3] = (v_eval[i] - v_init[i] - RHS_eval_v[i]*dt);
    }
    for (int i=0; i<6; ++i) {res[i] /= scales[i];}
    resnorm = 0.0;
    for (int i=0; i<ndim; ++i)
      resnorm += SQR(res[i]);
    resold = resnorm;

    Real x_step, v_step;
    Real x_fac, v_fac;
    x_fac = 1.0;
    v_fac = 1.0;

    // Start iterating
    // Using Newton method, thus computing the Jacobian at each iteration
    while( n_iter < it_max && resnorm > it_tol ){

    out_of_bounds = false; // Reset check variables
    invert_mat_fail = false;
    ++n_iter;
    for (int i = 0; i<3; ++i) {
      x_mid[i] = 0.5*(x_eval[i] + x_init[i]);
      v_mid[i] = 0.5*(v_eval[i] + v_init[i]);
    }
      
    // Position
    for (int dir = 0; dir<3; ++dir) {
      Real dirdx;
      int full_idx = dir;
      if (dir == 0) { i1 = 1; i2 = 2; dirdx = mbsize.d_view(m).dx1; }
      else if (dir == 1) { i1 = 0; i2 = 2; dirdx = mbsize.d_view(m).dx2; }
      else if (dir == 2) { i1 = 0; i2 = 1; dirdx = mbsize.d_view(m).dx3; }
      x_step = std::fmax(base_x_step, std::abs(x_init[dir]*x_fac*base_x_step));
      x_step = std::fmin(x_step, dirdx);

      x_grad[dir] = 0.5*(x_init[dir] + x_eval[dir]) + x_step;
      x_grad[i1] = x_mid[i1]; x_grad[i2] = x_mid[i2];
      GRRHSPosition(x_grad, v_mid, is_minkowski, spin, RHS_grad_x1);
      GRRHSVelocity(x_grad, v_mid, is_minkowski, spin, RHS_grad_v1);
      InterpolateFields(x_grad, b0_, e0_, mbsize, indcs, m, E, B, out_of_bounds);
      if (out_of_bounds) {break;}
      GRLorentz_Terms(x_grad, v_mid, E, B, is_minkowski, spin, q_over_m, RHS_grad_v1);
      x_grad[dir] = 0.5*(x_init[dir] + x_eval[dir]) - x_step;
      GRRHSPosition(x_grad, v_mid, is_minkowski, spin, RHS_grad_x2);
      GRRHSVelocity(x_grad, v_mid, is_minkowski, spin, RHS_grad_v2);
      InterpolateFields(x_grad, b0_, e0_, mbsize, indcs, m, E, B, out_of_bounds);
      if (out_of_bounds) {break;}
      GRLorentz_Terms(x_grad, v_mid, E, B, is_minkowski, spin, q_over_m, RHS_grad_v2);
      for (int i=0; i<3; ++i) { // Here Jacobian is for full system, position + velocity
        Jacob[full_idx][i] = -(RHS_grad_x1[i] - RHS_grad_x2[i])*dt/(2.0*x_step);
        Jacob[full_idx][i+3] = -(RHS_grad_v1[i] - RHS_grad_v2[i])*dt/(2.0*x_step);
      }
      for (int i=0; i<6; ++i) {Jacob[full_idx][i] /= scales[i];}
      Jacob[full_idx][full_idx] += 1.0 + base_x_step;  // Diagonal terms with Tikhonov
    }

    if (!out_of_bounds) {
      InterpolateFields(x_mid, b0_, e0_, mbsize, indcs, m, E, B, out_of_bounds);
    }

    if (!out_of_bounds) {
      // Velocity
      for (int dir = 0; dir<3; ++dir) {
        int full_idx = dir + 3;
        if (dir == 0) { i1 = 1; i2 = 2; }
        else if (dir == 1) { i1 = 0; i2 = 2; }
        else if (dir == 2) { i1 = 0; i2 = 1; }
        v_step = std::fmax(base_v_step, std::abs(v_init[dir]*v_fac*base_v_step));
        v_step = std::fmin(v_step, std::abs(v_init[dir]*0.1));

        v_grad[dir] = 0.5*(v_init[dir] + v_eval[dir]) + v_step;
        v_grad[i1] = v_mid[i1]; v_grad[i2] = v_mid[i2];
        GRRHSPosition(x_mid, v_grad, is_minkowski, spin, RHS_grad_x1);
        GRRHSVelocity(x_mid, v_grad, is_minkowski, spin, RHS_grad_v1);
        GRLorentz_Terms(x_mid, v_grad, E, B, is_minkowski, spin, q_over_m, RHS_grad_v1);
        v_grad[dir] = 0.5*(v_init[dir] + v_eval[dir]) - v_step;
        GRRHSPosition(x_mid, v_grad, is_minkowski, spin, RHS_grad_x2);
        GRRHSVelocity(x_mid, v_grad, is_minkowski, spin, RHS_grad_v2);
        GRLorentz_Terms(x_mid, v_grad, E, B, is_minkowski, spin, q_over_m, RHS_grad_v2);
        for (int i=0; i<3; ++i) { // Here Jacobian is for full system, position + velocity
          Jacob[full_idx][i] = -(RHS_grad_x1[i] - RHS_grad_x2[i])*dt/(2.0*v_step);
          Jacob[full_idx][i+3] = -(RHS_grad_v1[i] - RHS_grad_v2[i])*dt/(2.0*v_step);
        }
        for (int i=0; i<6; ++i) {Jacob[full_idx][i] /= scales[i];}
        Jacob[full_idx][full_idx] += 1.0 + base_v_step; // Diagonal terms with Tikhonov
      }
    
      // This not ideal: Size of matrix is fixed
      for (int ii = 0; ii<ndim; ++ii){
        for (int ij = 0; ij<3; ++ij){
          Jacob1D[ii*ndim + ij] = Jacob[ii][ij];
        }
        for (int ij = 3; ij<6; ++ij){
          Jacob1D[ii*ndim + ij] = Jacob[ii][ij];
        }
      }
      InvertMatrixLU( ndim, Jacob1D, invJacob1D, invert_mat_fail );
      if (invert_mat_fail) {
        x_fac *= 1.5 ;
        v_fac *= 1.5;
        continue;
      }

      GRRHSPosition(x_mid, v_mid, is_minkowski, spin, RHS_eval_x);
      GRRHSVelocity(x_mid, v_mid, is_minkowski, spin, RHS_eval_v);
      // InterpolateFields( pi(PTAG,p), x_eval, b0_, e0_, mbsize, indcs, m, E, B );
      GRLorentz_Terms(x_mid, v_mid, E, B, is_minkowski, spin, q_over_m, RHS_eval_v);
      for (int i=0; i<3; ++i) {
        res[i]   = (x_eval[i] - x_init[i] - RHS_eval_x[i]*dt);
        res[i+3] = (v_eval[i] - v_init[i] - RHS_eval_v[i]*dt);
      }
      for (int i=0; i<6; ++i) {res[i] /= scales[i];}
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

    resnorm = 0.0;
    for (int i = 0; i<ndim; ++i) { resnorm += SQR(res[i]); }

    if (resnorm < resold) {
      // Regular iteration
      for (int i=0; i<3; ++i){
        scaled_update_x[i] = 0.0;
        scaled_update_v[i] = 0.0;
        for (int j=0; j<6; ++j){
          scaled_update_x[i] -= invJacob1D[i*ndim + j]*res[j];
          scaled_update_v[i] -= invJacob1D[(i+3)*ndim + j]*res[j];
        }
        x_eval[i] += scaled_update_x[i]*scales[i];
        v_eval[i] += scaled_update_v[i]*scales[i+3];
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
          scaled_update_x[i] = 0.0;
          scaled_update_v[i] = 0.0;
          for (int j=0; j<6; ++j){
            scaled_update_x[i] -= damp_f*invJacob1D[i*ndim + j]*res[j];
            scaled_update_v[i] -= damp_f*invJacob1D[(i+3)*ndim + j]*res[j];
          }
          x_eval[i] += scaled_update_x[i]*scales[i];
          v_eval[i] += scaled_update_v[i]*scales[i+3];
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

    }; // while

    // Done with iterations, update ``true'' values
    if (n_iter == it_max) {
      // Fallback to mid-point explicit update
      aux_n_fails++;
      GRRHSPosition(x_init, v_init, is_minkowski, spin, RHS_eval_x);
      GRRHSVelocity(x_init, v_init, is_minkowski, spin, RHS_eval_v);
      InterpolateFields( x_init, b0_, e0_, mbsize, indcs, m, E, B, out_of_bounds );
      GRLorentz_Terms(x_init, v_init, E, B, is_minkowski, spin, q_over_m, RHS_eval_v);
      for (int i = 0; i<3; ++i) {
        x_mid[i] = x_init[i] + 0.5*dt*(RHS_eval_x[i]);
        v_mid[i] = v_init[i] + 0.5*dt*(RHS_eval_v[i]);
      }
      GRRHSPosition(x_mid, v_mid, is_minkowski, spin, RHS_eval_x);
      GRRHSVelocity(x_mid, v_mid, is_minkowski, spin, RHS_eval_v);
      InterpolateFields( x_mid, b0_, e0_, mbsize, indcs, m, E, B, out_of_bounds );
      GRLorentz_Terms(x_mid, v_mid, E, B, is_minkowski, spin, q_over_m, RHS_eval_v);
      for (int i = 0; i<3; ++i) {
        x_eval[i] = x_init[i] + dt*(RHS_eval_x[i]);
        v_eval[i] = v_init[i] + dt*(RHS_eval_v[i]);
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
  fail_num += tot_n_fails;
  if (nprtcl_thispack > 0) { average_iteration_number = static_cast<Real>(avg_iter)/nprtcl_thispack; }
  else { 
    average_iteration_number = 0.0; 
    tot_max_iter = 0;
  }
  max_iteration_number = tot_max_iter;
  return;
}

} // namespace particles
