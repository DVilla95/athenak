//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file gr_torus.cpp
//! \brief Test problem that initializes a monopole as prescribed by Michel 1973 to check particles
//! accelerated as per theoretical prediction in GR+electromagnetic forces
//!
//! References:
//!    Fishbone & Moncrief 1976, ApJ 207 962 (FM)
//!    Fishbone 1977, ApJ 215 323 (F)
//!    Chakrabarti, S. 1985, ApJ 288, 1

#include <stdio.h>
#include <math.h>

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#include <algorithm>  // max(), max_element(), min(), min_element()
#include <iomanip>
#include <iostream>   // endl
#include <limits>     // numeric_limits::max()
#include <memory>
#include <sstream>    // stringstream
#include <string>     // c_str(), string
#include <vector>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "mhd/mhd.hpp"

#include "particles/particles.hpp"
#include "particles/hamiltonian_gr.hpp"

#include <Kokkos_Random.hpp>

// prototypes for functions used internally to this pgen
namespace {

KOKKOS_INLINE_FUNCTION
static Real CalculateCovariantUT(Real spin, Real r, Real sin_theta, Real l);

KOKKOS_INLINE_FUNCTION
static void TransformVector(Real spin,
                            Real a0_bl, Real a1_bl, Real a2_bl, Real a3_bl,
                            Real x1, Real x2, Real x3,
                            Real *pa0, Real *pa1, Real *pa2, Real *pa3);

KOKKOS_INLINE_FUNCTION
static Real A0(Real dpl_mom, Real a_bh, Real x1, Real x2, Real x3);
KOKKOS_INLINE_FUNCTION
static Real A3(Real dpl_mom, Real a_bh, Real x1, Real x2, Real x3);

KOKKOS_INLINE_FUNCTION
void EnergyConservationTest(HistoryData *hdata, Mesh *pm);

Real dpl_mom;
bool mnpl;

} // namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//! \brief Sets initial conditions for either Fishbone-Moncrief or Chakrabarti torus in GR
//! Compile with '-D PROBLEM=gr_torus' to enroll as user-specific problem generator
//!  assumes x3 is axisymmetric direction
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  user_hist_func = EnergyConservationTest;

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int ie = indcs.ie, je = indcs.je, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  auto &coord = pmbp->pcoord->coord_data;

  // Extract BH parameters
  const Real r_excise = coord.rexcise;
  const bool is_radiation_enabled = (pmbp->prad != nullptr);
  const bool monopole = pin->GetOrAddBoolean("problem", "monopole", false); 
  mnpl = monopole;
  const Real B0 = pin->GetOrAddReal("problem", "b0_strength", 1.0);

  Real dpl = 0.0;
  if (!monopole) {
    dpl = pin->GetReal("problem", "dipole_moment");
    dpl_mom = dpl;
  }

  const bool has_particles = (pmbp->ppart != nullptr);
  // Need to split checks, otherwise a "particles" block will get initialized and written to the restart file
  // Then when trying to restart the mhd part this gives error because it looks for necessary particles parameters
  if (has_particles) {
    const bool inject_particles = pin->GetOrAddBoolean("particles", "inject_at_restart", false);
    auto &size = pmbp->pmb->mb_size;
    auto &indcs = pmbp->pmesh->mb_indcs;
    int &ng = indcs.ng;
    int n1 = indcs.nx1 + 2*ng;
    int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
    int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
    const int is = indcs.is;
    const int js = indcs.js;
    const int ks = indcs.ks;
    auto &gids = pmbp->gids;
    auto &gide = pmbp->gide;
    auto &coord = pmbp->pcoord->coord_data;
    int &npart = pmbp->ppart->nprtcl_thispack;
    auto &pr = pmbp->ppart->prtcl_rdata;
    auto &pi = pmbp->ppart->prtcl_idata;
    Real massive = 1.0; //This should be based on ptype, not hard-coded
    Real min_en = pin->GetOrAddReal("problem", "prtcl_energy_min", 1.005);
    Real max_en = pin->GetOrAddReal("problem", "prtcl_energy_max", 1.5);
    std::string prtcl_init_type = pin->GetString("particles","init_type");
    const Real q_over_m = pin->GetOrAddReal("particles", "charge_over_mass", 1);
    const bool is_gca = pmbp->ppart->is_gca;

    // Define criterium for how to initialize particles
    Real min_rad = pmbp->ppart->min_radius;
    Real crit, crit_min;

    // Initialize particles within a specific spherical shell
    if ( ! pin->DoesParameterExist("particles", "r_init_max") ) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "Particle initialization type " << prtcl_init_type <<" missing required parameter: " << "r_init_max" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    crit = pin->GetReal("particles", "r_init_max");
    crit_min = pin->GetOrAddReal("particles", "r_init_min", 0.0); 
    min_rad = fmax( min_rad, crit_min );

    if (inject_particles) {
      // Define criterium for how to initialize particles
      int num_good_cells;
      int nmb = (pmbp->nmb_thispack);
      int &ng = indcs.ng;
      int n1 = indcs.nx1 + 2*ng;
      int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
      int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
      // Array stores whether or not the Cell is viable for particle injection
      // based on criterium
      DvceArray2D<int> only_good_cells; // Array holds the m,k,j,i indeces of the cells that can be used to initialize particles
      pmbp->ppart->SelectCellsForInjection(only_good_cells, num_good_cells);

      if ( num_good_cells==0 ) {
        auto &pr = pmbp->ppart->prtcl_rdata;
        auto &pi = pmbp->ppart->prtcl_idata;
        int &npart = pmbp->ppart->nprtcl_thispack;
        npart = 0;
        Kokkos::realloc(pr, pmbp->ppart->nrdata, 0);
        Kokkos::realloc(pi, pmbp->ppart->nidata, 0);
        std::cout << "None of the MBs on rank " << global_variable::my_rank << " satisfy the injection criterium. Deleted particles."<< std::endl;
      } else {
        Kokkos::Random_XorShift64_Pool<> prtcl_rand(gids);
        Real L = pin->GetOrAddReal("particles", "angular_momentum", 0.0); 
        Real E = pin->GetOrAddReal("particles", "energy", 0.0); 
        Real chi = pin->GetOrAddReal("particles", "chi", 0.0); 

        par_for("part_init", DevExeSpace(),0,(npart-1),
          KOKKOS_LAMBDA(const int p){
            bool found_mb = false;
            auto prtcl_gen = prtcl_rand.get_state();
            const int cc = static_cast<int>(prtcl_gen.frand()*num_good_cells);
            const int m = only_good_cells(cc,0);
            const int k = only_good_cells(cc,1);
            const int j = only_good_cells(cc,2);
            const int i = only_good_cells(cc,3);
            const Real x1min = size.d_view(m).x1min;
            const Real x1max = size.d_view(m).x1max;
            const Real x2min = size.d_view(m).x2min;
            const Real x2max = size.d_view(m).x2max;
            const Real x3min = size.d_view(m).x3min;
            const Real x3max = size.d_view(m).x3max;
            Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
            Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
            Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
            Real r, th, phi;
            GetBoyerLindquistCoordinates(coord.bh_spin, x1v, x2v, x3v, &r, &th, &phi);
            if (monopole){
              if (!is_gca) {
                pr(IPVX,p) = 0.0; 	
                pr(IPVY,p) = 0.0; 	
                pr(IPVZ,p) = 0.0; 	
              } else {
                // For GCA the prtcl_energy_max actually acts to set the gamma factor/energy
                pr(IPVX,p) = sqrt(max_en);
                pr(IPVY,p) = min_en;
              }
            } else {
              if (!is_gca) {
                // Get components of velocity based on ang momentum and energy
                // Covariant vectors in BL coordinates
                Real cos2 = SQR(cos(th));
                Real sin2 = SQR(sin(th));
                Real r2 = SQR(r);
                Real asqr = SQR(coord.bh_spin);
                Real dd = (r2+asqr*cos2);
                Real delta = r2 - 2.0*r + asqr;
                Real A = SQR(r2 + asqr) - asqr*delta*sin2;
                // Upper metric components
                Real g00 = -(r2+asqr+2.0*r*asqr*sin2/dd)/delta;
                Real g11 = delta/dd;
                Real g22 = 1.0/dd;
                Real g33 = (delta-asqr*sin2)/(dd*delta*sin2);
                Real g03 = -2.0*r*coord.bh_spin/(delta*dd);
                // Lower metric components
                Real g_00 = -(1.0 - 2.0*r/dd);
                Real g_11 = 1.0/g11;
                Real g_22 = 1.0/g22;
                Real g_33 = sin2*(r2 + asqr + sin2*2.0*r*asqr/dd);
                Real g_03 = -2.0*r*coord.bh_spin*sin2/dd;

                Real u_0 = - E - q_over_m*A0(dpl, coord.bh_spin, x1v, x2v, x3v);
                Real u_ph = L - q_over_m*A3(dpl, coord.bh_spin, x1v, x2v, x3v);
                // Raise indeces
                Real tmp0, tmp1, tmp2, tmp3;
                tmp0 = g00*u_0 + g03*u_ph;
                tmp3 = g03*u_0 + g33*u_ph;
                Real u_r = 0.0;
                Real u_th = 0.0;
                Real t_chi = tan(chi);
                // These are upper components
                u_r = sqrt( (-1.0 - SQR(tmp0)*g_00 - 2.0*tmp0*tmp3*g_03 - SQR(tmp3)*g_33) / (g_11 + SQR(t_chi)*g_22) );
                u_th = t_chi*u_r;
                
                // Convert to cartesian
                Real uc0, uc1, uc2, uc3;
                TransformVector(coord.bh_spin, tmp0, u_r, u_th, tmp3,
                      x1v, x2v, x3v, &uc0, &uc1, &uc2, &uc3);
                
                Real gu[4][4], gl[4][4];
                ComputeMetricAndInverse(x1v,x2v,x3v, coord.is_minkowski, coord.bh_spin, gl, gu); 
                tmp0 = gl[0][0]*uc0 + gl[0][1]*uc1 + gl[0][2]*uc2 + gl[0][3]*uc3;
                tmp1 = gl[1][0]*uc0 + gl[1][1]*uc1 + gl[1][2]*uc2 + gl[1][3]*uc3;
                tmp2 = gl[2][0]*uc0 + gl[2][1]*uc1 + gl[2][2]*uc2 + gl[2][3]*uc3;
                tmp3 = gl[3][0]*uc0 + gl[3][1]*uc1 + gl[3][2]*uc2 + gl[3][3]*uc3;

                // Store lower cartesian components
                pr(IPVX,p) = tmp1;
                pr(IPVY,p) = tmp2; 	
                pr(IPVZ,p) = tmp3; 	
              } else {
                // For GCA the prtcl_energy_max actually acts to set the gamma factor/energy
                pr(IPVX,p) = sqrt(max_en);
                pr(IPVY,p) = min_en;
              }
            }
            pi(PGID,p) = gids+m;
            pr(IPX,p) = x1v;
            pr(IPY,p) = x2v;
            pr(IPZ,p) = x3v;
            prtcl_rand.free_state(prtcl_gen);
        });
        std::cout << "Injected " << npart << " particles." << std::endl;
      }
    }
    pmbp->pmesh->UpdatePrtclInfo();
  }
  pmbp->pmesh->dt = 1.0; // Hard-coded

  // return if restart
  if (restart) return;

  // Select either Hydro or MHD
  DvceArray5D<Real> u0_, w0_;
  u0_ = pmbp->pmhd->u0;
  w0_ = pmbp->pmhd->w0;

  // initialize primitive variables for new run ---------------------------------------
  Real r_s = 3.0;
  Real R_LC = 10.0*r_s;
  auto &size = pmbp->pmb->mb_size;
  auto &b0 = pmbp->pmhd->b0;
  auto &e0 = pmbp->pmhd->efld;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;

  if (monopole) {
    par_for("pgen_monopole", DevExeSpace(),0,(nmb-1),ks,ke+1,js,je+1,is,ie+1,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {

      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real x1v = LeftEdgeX(i-is, indcs.nx1, x1min, x1max);
      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real x2v = LeftEdgeX(j-js, indcs.nx2, x2min, x2max);
      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real x3v = LeftEdgeX(k-ks, indcs.nx3, x3min, x3max);

      // Extract metric and inverse
      Real glower[4][4], gupper[4][4];
      ComputeMetricAndInverse(x1v, x2v, x3v, coord.is_minkowski, coord.bh_spin,
                              glower, gupper);

      // Calculate spherical coordinates of cell
      Real r, theta, phi;
      r = sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));
      phi = atan2(x2v, x1v);
      theta = atan2(sqrt(SQR(x1v)+SQR(x2v)), x3v);
      // GetBoyerLindquistCoordinates(coord.bh_spin, x1v, x2v, x3v, &r, &theta, &phi);
      Real sin_theta = sin(theta);
      Real cos_theta = cos(theta);
      Real sin_phi = sin(phi);
      Real cos_phi = cos(phi);

      // Calculate background primitives
      Real rho_min = pin->GetReal("problem", "rho_min");
      Real rho_pow = pin->GetReal("problem", "rho_pow");
      Real pgas_min = pin->GetReal("problem", "pgas_min");
      Real pgas_pow = pin->GetReal("problem", "pgas_pow");
      Real rho_bg, pgas_bg;
      if (r > 1.0) {
        rho_bg = rho_min * pow(r, rho_pow);
        pgas_bg = pgas_min * pow(r, pgas_pow);
      } else {
        rho_bg = coord.dexcise;
        pgas_bg = coord.pexcise;
      }

      Real rho = rho_bg;
      Real pgas = pgas_bg;
      Real uu1 = 0.0;
      Real uu2 = 0.0;
      Real uu3 = 0.0;

      // Set primitive values, including random perturbations to pressure
      w0_(m,IDN,k,j,i) = fmax(rho, rho_bg);
      w0_(m,IEN,k,j,i) = fmax(pgas, pgas_bg);
      w0_(m,IVX,k,j,i) = uu1;
      w0_(m,IVY,k,j,i) = uu2;
      w0_(m,IVZ,k,j,i) = uu3;

      if (r < 1.0) {
        b0.x1f(m,k,j,i) = 0.0;
        b0.x2f(m,k,j,i) = 0.0;
        b0.x3f(m,k,j,i) = 0.0;
        e0.x1e(m,k,j,i) = 0.0;
        e0.x2e(m,k,j,i) = 0.0;
        e0.x3e(m,k,j,i) = 0.0;
      } else {
        Real D_theta = -B0*(r_s/R_LC)*(r_s/r)*sin_theta;
        Real B_phi = D_theta;
        Real B_r = B0*(r_s/r)*(r_s/r);
        Real B_x, B_y, B_z;
        Real E_x, E_y, E_z;
      
        B_x = cos_phi*sin_theta*B_r - sin_phi*B_phi;
        B_y = sin_phi*sin_theta*B_r + cos_phi*B_phi;
        B_z = cos_theta*B_r;
        b0.x1f(m,k,j,i) = B_x;
        b0.x2f(m,k,j,i) = B_y;
        b0.x3f(m,k,j,i) = B_z;

        E_x = cos_theta*cos_phi*D_theta;
        E_y = cos_theta*sin_phi*D_theta;
        E_z = -sin_theta*D_theta;
        Real adm_det; 
        Real adm[3][3];
        GetUpperAdmMetric( gupper, adm );
        ComputeDeterminant3( adm, adm_det );
        //Compute ExB drift 
        // First need to convert electric field to normal frame
        // Vector product results in covariant vector
        Real E_beta[3] = {
          - gupper[0][2]/gupper[0][0]*B_z + B_y*gupper[0][3]/gupper[0][0],
          - gupper[0][3]/gupper[0][0]*B_x + B_z*gupper[0][1]/gupper[0][0],
          - gupper[0][1]/gupper[0][0]*B_y + B_x*gupper[0][2]/gupper[0][0]
        };
        //for (int i = 0; i < 3; ++i ){ E_beta[i] /= adm_det; }
        Real E_aux[3] = {0.0};
        E_aux[0] = adm[0][0]*E_beta[0] + adm[0][1]*E_beta[1] + adm[0][2]*E_beta[2];
        E_aux[1] = adm[1][0]*E_beta[0] + adm[1][1]*E_beta[1] + adm[1][2]*E_beta[2];
        E_aux[2] = adm[2][0]*E_beta[0] + adm[2][1]*E_beta[1] + adm[2][2]*E_beta[2];
        Real alpha = sqrt(-1/gupper[0][0]); 
        E_x = E_x*alpha + E_aux[0]*adm_det;
        E_y = E_y*alpha + E_aux[1]*adm_det;
        E_z = E_z*alpha + E_aux[2]*adm_det;

        e0.x1e(m,k,j,i) = E_x;
        e0.x2e(m,k,j,i) = E_y;
        e0.x3e(m,k,j,i) = E_z;

        if (i==ie) {
         b0.x1f(m,k,j,i+1) = B_x;
         e0.x1e(m,k,j,i+1) = E_x;
        }
        if (j==je) {
          b0.x2f(m,k,j+1,i) = B_y;
          e0.x2e(m,k,j+1,i) = E_y;
        }
        if (k==ke) {
          b0.x3f(m,k+1,j,i) = B_z;
          e0.x3e(m,k+1,j,i) = E_z;
        }
      }
    });
  } else { // not monopole, i.e. dipole
    DvceArray4D<Real> a0, a1, a2, a3;
    Kokkos::realloc(a0,nmb,n3,n2,n1);
    Kokkos::realloc(a1,nmb,n3,n2,n1);
    Kokkos::realloc(a2,nmb,n3,n2,n1);
    Kokkos::realloc(a3,nmb,n3,n2,n1);

    par_for("pgen_dipole", DevExeSpace(),0,(nmb-1),ks,ke+1,js,je+1,is,ie+1,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {

      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

      // A0 and A3 are given in spherical BL
      Real r, theta, phi;
      GetBoyerLindquistCoordinates(coord.bh_spin, x1v, x2v, x3v, &r, &theta, &phi);
      Real cos2 = SQR(cos(theta));
      Real sin2 = SQR(sin(theta));
      Real r2 = SQR(r);
      Real asqr = SQR(coord.bh_spin);
      Real dd = (r2+asqr*cos2);
      Real delta = r2 - 2.0*r + asqr;
      Real A = SQR(r2 + asqr) - asqr*delta*sin2;
      // Upper metric components
      Real g00 = -(r2+asqr+2.0*r*asqr*sin2/dd)/delta;
      Real g11 = delta/dd;
      Real g22 = 1.0/dd;
      Real g33 = (delta-asqr*sin2)/(dd*delta*sin2);
      Real g03 = -2.0*r*coord.bh_spin/(delta*dd);
      // Raise indeces on potential 4-vector for use with TransformVector
      Real tmp0 = g00*A0(dpl, coord.bh_spin, x1v, x2v, x3v) + g03*A3(dpl, coord.bh_spin, x1v, x2v, x3v);
      Real tmp1 = 0.0;
      Real tmp2 = 0.0;
      Real tmp3 = g03*A0(dpl, coord.bh_spin, x1v, x2v, x3v) + g33*A3(dpl, coord.bh_spin, x1v, x2v, x3v);
      Real uc0, uc1, uc2, uc3;
      TransformVector(coord.bh_spin, tmp0, tmp1, tmp2, tmp3,
            x1v, x2v, x3v, &uc0, &uc1, &uc2, &uc3);
	    Real gu[4][4], gl[4][4];
	    ComputeMetricAndInverse(x1v,x2v,x3v, coord.is_minkowski, coord.bh_spin, gl, gu); 
      // Store covariant 4-potential vector
      a0(m,k,j,i) = gl[0][0]*uc0 + gl[0][1]*uc1 + gl[0][2]*uc2 + gl[0][3]*uc3;
      a1(m,k,j,i) = gl[1][0]*uc0 + gl[1][1]*uc1 + gl[1][2]*uc2 + gl[1][3]*uc3;
      a2(m,k,j,i) = gl[2][0]*uc0 + gl[2][1]*uc1 + gl[2][2]*uc2 + gl[2][3]*uc3;
      a3(m,k,j,i) = gl[3][0]*uc0 + gl[3][1]*uc1 + gl[3][2]*uc2 + gl[3][3]*uc3;

      r = sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));
      phi = atan2(x2v, x1v);
      theta = atan2(sqrt(SQR(x1v)+SQR(x2v)), x3v);
      // Calculate background primitives
      Real rho_min = pin->GetReal("problem", "rho_min");
      Real rho_pow = pin->GetReal("problem", "rho_pow");
      Real pgas_min = pin->GetReal("problem", "pgas_min");
      Real pgas_pow = pin->GetReal("problem", "pgas_pow");
      Real rho_bg, pgas_bg;
      if (r > 1.0) {
        rho_bg = rho_min * pow(r, rho_pow);
        pgas_bg = pgas_min * pow(r, pgas_pow);
      } else {
        rho_bg = coord.dexcise;
        pgas_bg = coord.pexcise;
      }

      Real rho = rho_bg;
      Real pgas = pgas_bg;
      Real uu1 = 0.0;
      Real uu2 = 0.0;
      Real uu3 = 0.0;

      // Set primitive values, including random perturbations to pressure
      w0_(m,IDN,k,j,i) = fmax(rho, rho_bg);
      w0_(m,IEN,k,j,i) = fmax(pgas, pgas_bg);
      w0_(m,IVX,k,j,i) = uu1;
      w0_(m,IVY,k,j,i) = uu2;
      w0_(m,IVZ,k,j,i) = uu3;
    });

    par_for("pgen_b0", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real x1v = LeftEdgeX(i-is, indcs.nx1, x1min, x1max);
      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real x2v = LeftEdgeX(j-js, indcs.nx2, x2min, x2max);
      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real x3v = LeftEdgeX(k-ks, indcs.nx3, x3min, x3max);

      // Calculate spherical coordinates of cell
      Real r;
      r = sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));
      if (r < 0.5) {
        b0.x1f(m,k,j,i) = 0.0;
        b0.x2f(m,k,j,i) = 0.0;
        b0.x3f(m,k,j,i) = 0.0;
        e0.x1e(m,k,j,i) = 0.0;
        e0.x2e(m,k,j,i) = 0.0;
        e0.x3e(m,k,j,i) = 0.0;
      } else {
        // Compute face-centered fields from curl(A).
        Real dx1 = size.d_view(m).dx1;
        Real dx2 = size.d_view(m).dx2;
        Real dx3 = size.d_view(m).dx3;

        // Get components F_ij
        // f13 has already opposite sign
        Real f23 = ((a3(m,k,j+1,i) - a3(m,k,j,i))/dx2 -
                   (a2(m,k+1,j,i) - a2(m,k,j,i))/dx3);
        Real f13 = ((a1(m,k+1,j,i) - a1(m,k,j,i))/dx3 -
                   (a3(m,k,j,i+1) - a3(m,k,j,i))/dx1);
        Real f12 = ((a2(m,k,j,i+1) - a2(m,k,j,i))/dx1 -
                   (a1(m,k,j+1,i) - a1(m,k,j,i))/dx2);

        // Store B
        // This is consistent with AthenaK storing *F^0i
        b0.x1f(m,k,j,i) = f23;
        b0.x2f(m,k,j,i) = f13; // This has already opposite sign
        b0.x3f(m,k,j,i) = f12;

        // Get components F_i0
        Real f01 = (a0(m,k,j,i+1) - a0(m,k,j,i))/dx1;
        Real f02 = (a0(m,k,j+1,i) - a0(m,k,j,i))/dx2;
        Real f03 = (a0(m,k+1,j,i) - a0(m,k,j,i))/dx3;

        // Store E
        // This is consistent with AthenaK storing *F^jk
        e0.x1e(m,k,j,i) = f01;
        e0.x2e(m,k,j,i) = f02;
        e0.x3e(m,k,j,i) = f03;

        // Include extra face-component at edge of block in each direction
        if (i==ie) {
          f01 = (a0(m,k,j,i+2) - a0(m,k,j,i+1))/dx1;
          f02 = (a0(m,k,j+1,i+1) - a0(m,k,j,i+1))/dx2;
          f03 = (a0(m,k+1,j,i+1) - a0(m,k,j,i+1))/dx3;

          f23 = ((a3(m,k,j+1,i+1) - a3(m,k,j,i+1))/dx2 -
                (a2(m,k+1,j,i+1) - a2(m,k,j,i+1))/dx3);
          f13 = ((a1(m,k+1,j,i+1) - a1(m,k,j,i+1))/dx3 -
                (a3(m,k,j,i+2) - a3(m,k,j,i+1))/dx1);
          f12 = ((a2(m,k,j,i+2) - a2(m,k,j,i+1))/dx1 -
                (a1(m,k,j+1,i+1) - a1(m,k,j,i+1))/dx2);

          e0.x1e(m,k,j,i+1) = f01;
          e0.x2e(m,k,j,i+1) = f02;
          e0.x3e(m,k,j,i+1) = f03;
          b0.x1f(m,k,j,i+1) = f23;
          b0.x2f(m,k,j,i+1) = f13;
          b0.x3f(m,k,j,i+1) = f12;
        }
        if (j==je) {
          f01 = (a0(m,k,j+1,i+1) - a0(m,k,j+1,i))/dx1;
          f02 = (a0(m,k,j+2,i) - a0(m,k,j+1,i))/dx2;
          f03 = (a0(m,k+1,j+1,i) - a0(m,k,j+1,i))/dx3;

          f23 = ((a3(m,k,j+2,i) - a3(m,k,j+1,i))/dx2 -
                (a2(m,k+1,j+1,i) - a2(m,k,j+1,i))/dx3);
          f13 = ((a1(m,k+1,j+1,i) - a1(m,k,j+1,i))/dx3 -
                (a3(m,k,j+1,i+1) - a3(m,k,j+1,i))/dx1);
          f12 = ((a2(m,k,j+1,i+1) - a2(m,k,j+1,i))/dx1 -
                (a1(m,k,j+2,i) - a1(m,k,j+1,i))/dx2);

          e0.x1e(m,k,j+1,i) = f01;
          e0.x2e(m,k,j+1,i) = f02;
          e0.x3e(m,k,j+1,i) = f03;
          b0.x1f(m,k,j+1,i) = f23;
          b0.x2f(m,k,j+1,i) = f13;
          b0.x3f(m,k,j+1,i) = f12;
        }
        if (k==ke) {
          f01 = (a0(m,k+1,j,i+1) - a0(m,k+1,j,i))/dx1;
          f02 = (a0(m,k+1,j+1,i) - a0(m,k+1,j,i))/dx2;
          f03 = (a0(m,k+2,j,i) - a0(m,k+1,j,i))/dx3;

          f23 = ((a3(m,k+1,j+1,i) - a3(m,k+1,j,i))/dx2 -
                (a2(m,k+2,j,i) - a2(m,k+1,j,i))/dx3);
          f13 = ((a1(m,k+2,j,i) - a1(m,k+1,j,i))/dx3 -
                (a3(m,k+1,j,i+1) - a3(m,k+1,j,i))/dx1);
          f12 = ((a2(m,k+1,j,i+1) - a2(m,k+1,j,i))/dx1 -
                (a1(m,k+1,j+1,i) - a1(m,k+1,j,i))/dx2);

          e0.x1e(m,k+1,j,i) = f01;
          e0.x2e(m,k+1,j,i) = f02;
          e0.x3e(m,k+1,j,i) = f03;
          b0.x1f(m,k+1,j,i) = f23;
          b0.x2f(m,k+1,j,i) = f13;
          b0.x3f(m,k+1,j,i) = f12;
        }
      }
    });

  }

  // Compute cell-centered fields
  auto &bcc_ = pmbp->pmhd->bcc0;
  par_for("pgen_bcc", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // cell-centered fields are simple linear average of face-centered fields
    Real& w_bx = bcc_(m,IBX,k,j,i);
    Real& w_by = bcc_(m,IBY,k,j,i);
    Real& w_bz = bcc_(m,IBZ,k,j,i);
    w_bx = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1));
    w_by = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i));
    w_bz = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i));
  });
  // Convert primitives to conserved
  auto &bcc0_ = pmbp->pmhd->bcc0;
  pmbp->pmhd->peos->PrimToCons(w0_, bcc0_, u0_, is, ie, js, je, ks, ke);

  return;
}

namespace {

//----------------------------------------------------------------------------------------
// Function for transforming 4-vector from Boyer-Lindquist to desired coordinates
// Inputs:
//   a0_bl,a1_bl,a2_bl,a3_bl: upper 4-vector components in Boyer-Lindquist coordinates
//   x1,x2,x3: Cartesian Kerr-Schild coordinates of point
// Outputs:
//   pa0,pa1,pa2,pa3: pointers to upper 4-vector components in desired coordinates
// Notes:
//   Schwarzschild coordinates match Boyer-Lindquist when a = 0

KOKKOS_INLINE_FUNCTION
static void TransformVector(Real spin,
                            Real a0_bl, Real a1_bl, Real a2_bl, Real a3_bl,
                            Real x1, Real x2, Real x3,
                            Real *pa0, Real *pa1, Real *pa2, Real *pa3) {
  Real rad = sqrt( SQR(x1) + SQR(x2) + SQR(x3) );
  Real r = fmax((sqrt( SQR(rad) - SQR(spin) + sqrt(SQR(SQR(rad)-SQR(spin))
                      + 4.0*SQR(spin)*SQR(x3)) ) / sqrt(2.0)), 1.0);
  Real delta = SQR(r) - 2.0*r + SQR(spin);
  *pa0 = a0_bl + 2.0*r/delta * a1_bl;
  *pa1 = a1_bl * ( (r*x1+spin*x2)/(SQR(r) + SQR(spin)) - x2*spin/delta) +
         a2_bl * x1*x3/r * sqrt((SQR(r) + SQR(spin))/(SQR(x1) + SQR(x2))) -
         a3_bl * x2;
  *pa2 = a1_bl * ( (r*x2-spin*x1)/(SQR(r) + SQR(spin)) + x1*spin/delta) +
         a2_bl * x2*x3/r * sqrt((SQR(r) + SQR(spin))/(SQR(x1) + SQR(x2))) +
         a3_bl * x1;
  *pa3 = a1_bl * x3/r -
         a2_bl * r * sqrt((SQR(x1) + SQR(x2))/(SQR(r) + SQR(spin)));
  return;
}

//----------------------------------------------------------------------------------------
// Function to calculate time component of contravariant four velocity in BL
// Inputs:
//   r: radial Boyer-Lindquist coordinate
//   sin_theta: sine of polar Boyer-Lindquist coordinate
// Outputs:
//   returned value: u_t

KOKKOS_INLINE_FUNCTION
static Real CalculateCovariantUT(Real spin, Real r, Real sin_theta, Real l) {
  // Compute BL metric components
  Real sigma = SQR(r) + SQR(spin)*(1.0-SQR(sin_theta));
  Real g_00 = -1.0 + 2.0*r/sigma;
  Real g_03 = -2.0*spin*r/sigma*SQR(sin_theta);
  Real g_33 = (SQR(r) + SQR(spin) +
               2.0*SQR(spin)*r/sigma*SQR(sin_theta))*SQR(sin_theta);

  // Compute time component of covariant BL 4-velocity
  Real u_t = -sqrt(fmax((SQR(g_03) - g_00*g_33)/(g_33 + 2.0*l*g_03 + SQR(l)*g_00), 0.0));
  return u_t;
}

KOKKOS_INLINE_FUNCTION
static Real A0(Real dpl_mom, Real a_bh, Real x1, Real x2, Real x3) {
  // BL coordinates
  Real r, theta, phi;
  GetBoyerLindquistCoordinates(a_bh, x1, x2, x3, &r, &theta, &phi);

  Real m = 1.0;
  Real rp = m + sqrt(SQR(m) - SQR(a_bh));
  Real rm = m - sqrt(SQR(m) - SQR(a_bh));
  Real cos2 = SQR(cos(theta));
  Real sigma = SQR(r) + SQR(a_bh)*cos2;
  Real zeta = sqrt(SQR(m)-SQR(a_bh));
  Real prefac = (3.0/2.0)*(a_bh*dpl_mom)/( SQR(zeta) * sigma);
  Real p1 = r*(r - m) + (SQR(a_bh) - m*r)*cos2;

  return prefac*( p1*0.5/zeta*log((r-rm)/(r-rp)) - (r - m*cos2) );
}

KOKKOS_INLINE_FUNCTION
static Real A3(Real dpl_mom, Real a_bh, Real x1, Real x2, Real x3) {
  // BL coordinates
  Real r, theta, phi;
  GetBoyerLindquistCoordinates(a_bh, x1, x2, x3, &r, &theta, &phi);

  Real m = 1.0;
  Real rp = m + sqrt(SQR(m) - SQR(a_bh));
  Real rm = m - sqrt(SQR(m) - SQR(a_bh));
  Real delta = SQR(r) - 2.0*m*r + SQR(a_bh);
  Real cos2 = SQR(cos(theta));
  Real sigma = SQR(r) + SQR(a_bh)*cos2;
  Real zeta = sqrt(SQR(m)-SQR(a_bh));
  Real prefac = (3.0/4.0)*(SQR(sin(theta))*dpl_mom)/( SQR(zeta) * sigma);
  Real p1 = r*(r*r*r - 2.0*m*SQR(a_bh) + SQR(a_bh)*r) + delta*SQR(a_bh)*cos2;

  return prefac*( - (p1*0.5/zeta)*log((r-rm)/(r-rp)) + (r - m)*cos2*SQR(a_bh) + r*(SQR(r) + m*r + 2.0*SQR(a_bh)) );
}

void EnergyConservationTest(HistoryData *pdata, Mesh *pm){
	// This function computes the energy at time "curr_time" for each particle
	// and compares to the energy at the beginning of the simulation.
	// This is used as a metric for the accuracy of the gr integration.
	
	  // Re-compute the initial positions
	  auto &mbsize = pm->pmb_pack->pmb->mb_size;
	  auto &pr = pm->pmb_pack->ppart->prtcl_rdata;
	  auto &pi = pm->pmb_pack->ppart->prtcl_idata;
	  auto &npart = pm->pmb_pack->ppart->nprtcl_thispack;
	  auto gids = pm->pmb_pack->gids;
	  auto gide = pm->pmb_pack->gide;
    auto &coord = pm->pmb_pack->pcoord->coord_data;
	  const Real spin = coord.bh_spin;
	  const Real massive = 1.0; //TODO for photons/massless particles this needs to be 0: condition on ptype
	  const Real q_over_m = pm->pmb_pack->ppart->charge_over_mass;

    Real dpl = 0.0;
    if (!mnpl)
      dpl = dpl_mom;
	  pdata->nhist = npart;

    for (int p = 0; p<npart; ++p) {
      pdata->label[p] = std::to_string(static_cast<int>(pi(PTAG,p)));
    }

    for (int p = 0; p<npart; ++p) {
	    Real u[3] = {pr(IPVX,p), pr(IPVY,p), pr(IPVZ,p)};	  
	    Real x[3] = {pr(IPX,p), pr(IPY,p), pr(IPZ,p)};	  
	    Real gu[4][4], gl[4][4], adm[3][3];
	    ComputeMetricAndInverse(x[0],x[1],x[2], coord.is_minkowski, spin, gl, gu); 
      GetUpperAdmMetric( gu, adm );

      Real r, theta, phi;
      GetBoyerLindquistCoordinates(spin, x[0], x[1], x[2], &r, &theta, &phi);
      Real cos2 = SQR(cos(theta));
      Real sin2 = SQR(sin(theta));
      Real r2 = SQR(r);
      Real asqr = SQR(coord.bh_spin);
      Real dd = (r2+asqr*cos2);
      Real delta = r2 - 2.0*r + asqr;
      Real A = SQR(r2 + asqr) - asqr*delta*sin2;
      Real g00 = -A/(dd*delta);
      Real g11 = delta/dd;
      Real g22 = 1.0/dd;
      Real g33 = (delta-asqr*sin2)/(dd*delta*sin2);
      Real g03 = -2.0*r*coord.bh_spin/(delta*dd);
      Real tmp0 = g00*A0(dpl, spin, x[0], x[1], x[2]) + g03*A3(dpl, spin, x[0], x[1], x[2]);
      Real tmp1 = 0.0;
      Real tmp2 = 0.0;
      Real tmp3 = g03*A0(dpl, spin, x[0], x[1], x[2]) + g33*A3(dpl, spin, x[0], x[1], x[2]);
      Real uc0, uc1, uc2, uc3;
      TransformVector(spin, tmp0, tmp1, tmp2, tmp3,
            x[0], x[1], x[2], &uc0, &uc1, &uc2, &uc3);
      Real a0 = gl[0][0]*uc0 + gl[0][1]*uc1 + gl[0][2]*uc2 + gl[0][3]*uc3;
      Real a1 = gl[1][0]*uc0 + gl[1][1]*uc1 + gl[1][2]*uc2 + gl[1][3]*uc3;
      Real a2 = gl[2][0]*uc0 + gl[2][1]*uc1 + gl[2][2]*uc2 + gl[2][3]*uc3;
      Real a3 = gl[3][0]*uc0 + gl[3][1]*uc1 + gl[3][2]*uc2 + gl[3][3]*uc3;

      Real u0 = adm[0][0]*SQR(u[0]) + adm[1][1]*SQR(u[1]) + adm[2][2]*SQR(u[2])
        + 2.0*adm[0][1]*u[0]*u[1] + 2.0*adm[0][2]*u[0]*u[2] + 2.0*adm[1][2]*u[1]*u[2];
      u0 = sqrt(1.0 + u0)*sqrt(-gu[0][0]);
      Real u1, u2, u3;
      u1 = adm[0][0]*u[0] + adm[0][1]*u[1] + adm[0][2]*u[2];
      u1 = u1 + u0*gu[0][1]/gu[0][0]; //Subtract beta^i
      u2 = adm[1][0]*u[0] + adm[1][1]*u[1] + adm[1][2]*u[2];
      u2 = u2 + u0*gu[0][2]/gu[0][0]; //Subtract beta^i
      u3 = adm[2][0]*u[0] + adm[2][1]*u[1] + adm[2][2]*u[2];
      u3 = u3 + u0*gu[0][3]/gu[0][0]; //Subtract beta^i
      Real u_0 = gl[0][0]*u0 + gl[0][1]*u1 + gl[0][2]*u2 + gl[0][3]*u3;
      Real E = - u_0 - q_over_m*a0;
      pdata->hdata[p] = E;
    }
    for (int n=pdata->nhist; n<NHISTORY_VARIABLES; ++n) {
      pdata->hdata[n] = 0.0;
    }

	  return;
	  
}

} // namespace
