//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particles_injection.cpp
//! \brief implements functions for  particle injection. This
//! includes inline functions to get particle's velocity given background fluid properties,
//! functions to select candidate MBs for injection.

#include "athena.hpp"
#include "globals.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "particles/particles_helpers.hpp"
#include "diffusion/current_density.hpp"

#include <Kokkos_Random.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

KOKKOS_INLINE_FUNCTION
static Real ComputeAspectRatio( const int m, const int k, const int j, const int i,
          const int ke, const int je, const int ie, const Real j_thr , const Real ar_thr,
          const DvceArray5D<Real> &bcc_, const DvceArray5D<Real> &j_arr ) {
    Real npt_perp = 0.0;
    int k1 = k; int j1 = j; int i1 = i;
    // Loop only moves forwards
    while ( j_arr(m,IDN,k1,j1,i1) >= j_thr // absolute value above threshold
        &&  k1 <= ke && j1 <= je && i1 <= ie ) { // Don't leave meshblock
      npt_perp+=1.0;
      // Compute direction perpendicular to both B and J
      Real c1 = std::abs(
        bcc_(m,IBY,k1,j1,i1)*j_arr(m,IVZ,k1,j1,i1) - bcc_(m,IBZ,k1,j1,i1)*j_arr(m,IVY,k1,j1,i1)
      );
      Real c2 = std::abs(
        bcc_(m,IBZ,k1,j1,i1)*j_arr(m,IVX,k1,j1,i1) - bcc_(m,IBX,k1,j1,i1)*j_arr(m,IVZ,k1,j1,i1)
      );
      Real c3 = std::abs(
        bcc_(m,IBX,k1,j1,i1)*j_arr(m,IVY,k1,j1,i1) - bcc_(m,IBY,k1,j1,i1)*j_arr(m,IVX,k1,j1,i1)
      );
      // Move in "main" perpendicular direction
      if (c1 > c2) {
        if (c2 > c3) {i1++;}
        else {
          if (c1 > c3)  {i1++;}
          else          {k1++;}
        }
      } else {
        if (c1 > c3) {j1++;}
        else {
          if (c2 > c3)  {j1++;}
          else          {k1++;}
        }
      }
    }
    if (npt_perp < 4) {return -1.0;} // Position is of no interest
    Real npt_alng = 0.0;
    Real aspect_ratio = npt_alng/npt_perp;
    k1 = k; j1 = j; i1 = i;
    while ( j_arr(m,IDN,k1,j1,i1) >= j_thr
        &&  k1 <= ke && j1 <= je && i1 <= ie 
        && aspect_ratio < ar_thr ) { // If the aspect ratio is large enough stop early
      npt_alng+=1.0;
      aspect_ratio = npt_alng/npt_perp;
      Real c1 = std::abs(j_arr(m,IVX,k1,j1,i1));
      Real c2 = std::abs(j_arr(m,IVY,k1,j1,i1));
      Real c3 = std::abs(j_arr(m,IVZ,k1,j1,i1));
      if (c1 > c2) {
        if (c2 > c3) {i1++;}
        else {
          if (c1 > c3)  {i1++;}
          else          {k1++;}
        }
      } else {
        if (c1 > c3) {j1++;}
        else {
          if (c2 > c3)  {j1++;}
          else          {k1++;}
        }
      }
    }
    return aspect_ratio;
}

KOKKOS_INLINE_FUNCTION
static void ComputeCurrent( const Real j_thr, const Real thr_val, const int nmb, const CoordData &coord,
          const DvceFaceFld4D<Real> &bface, const DualArray1D<RegionSize> &size, const RegionIndcs &indcs,
          DvceArray5D<Real> &j_arr, Real &curr_max ) {
  const int nmkji = nmb*indcs.nx3*indcs.nx2*indcs.nx1;
  const int nkji = indcs.nx3*indcs.nx2*indcs.nx1;
  const int nji  = indcs.nx2*indcs.nx1;
  const int is = indcs.is;
  const int js = indcs.js;
  const int ks = indcs.ks;
  Kokkos::parallel_reduce("prtcls_computej", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &max_j) {
    // compute m,k,j,i indices of thread and call function
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/indcs.nx1;
    int i = (idx - m*nkji - k*nji - j*indcs.nx1) + is;
    k += ks;
    j += js;
    Real x1min = size.d_view(m).x1min;
    Real x1max = size.d_view(m).x1max;
    Real x2min = size.d_view(m).x2min;
    Real x2max = size.d_view(m).x2max;
    Real x3min = size.d_view(m).x3min;
    Real x3max = size.d_view(m).x3max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
    Real ju1 = 0.0; Real ju2 = 0.0; Real ju3 = 0.0; Real j_abs = 0.0;
    bool thr_check = true;
    CurlFC(m, k, j, i, bface, size.d_view(m), ju1, ju2, ju3, thr_val, thr_check);
    if (thr_check) {
      Real gu[4][4], gl[4][4];
      ComputeMetricAndInverse(x1v,x2v,x3v,coord.is_minkowski,coord.bh_spin,gl,gu); 
      Real alpha = sqrt(-1.0/gu[0][0]); // This is an approximation, should get alpha at both locations of B involved in curl
      Real m3[3][3];
      GetUpperAdmMetric( gu, m3 );
      // Determinant of metric needed for vector products
      Real m3_det; 
      ComputeDeterminant3( m3, m3_det );
      m3_det = 1.0/sqrt(m3_det);
      ju1 *= (m3_det*alpha); ju2 *= (m3_det*alpha); ju3 *= (m3_det*alpha);
      j_abs = gl[1][1]*SQR(ju1) + gl[2][2]*SQR(ju2) + gl[3][3]*SQR(ju3)
          + 2.0*gl[1][2]*ju1*ju2 + 2.0*gl[1][3]*ju1*ju3 + 2.0*gl[2][3]*ju2*ju3;
      j_abs = sqrt(j_abs);
    }
    j_arr(m,IVX,k,j,i) = ju1; j_arr(m,IVY,k,j,i) = ju2; j_arr(m,IVZ,k,j,i) = ju3;  
    j_arr(m,IDN,k,j,i) = j_abs;
    max_j = std::fmax(max_j,j_abs);
  }, Kokkos::Max<Real>(curr_max) );
  return;
}

namespace particles {

void Particles::SelectCellsForInjection(DvceArray2D<int> &only_good_cells, int &num_good_cells) {
  // Reset all MBs to false
  auto &coord = pmy_pack->pcoord->coord_data;
  auto &size  = pmy_pack->pmb->mb_size;
  const int nmb = pmy_pack->nmb_thispack;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int &ng = indcs.ng;
  const int n1 = indcs.nx1 + 2*ng;
  const int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  const int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  const int nmkji = nmb*indcs.nx3*indcs.nx2*indcs.nx1;
  const int nkji = indcs.nx3*indcs.nx2*indcs.nx1;
  const int nji  = indcs.nx2*indcs.nx1;
  const int is = indcs.is; const int ie = indcs.ie;
  const int js = indcs.js; const int je = indcs.je;
  const int ks = indcs.ks; const int ke = indcs.ke;

  DvceArray4D<bool> cells_for_injection;
  Kokkos::realloc(cells_for_injection, nmb, n3, n2, n1);

  DvceArray5D<Real> u0_, w0_;
  DvceArray5D<Real> bcc_, j_;
  Real curr_max;
  bool check_fluid = false;
  Real gamma_eos;
  check_fluid |= inject_pars.check_density || inject_pars.check_temperature;
  check_fluid |= inject_pars.check_current || inject_pars.check_beta;
  if (check_fluid) {
    if (pmy_pack->phydro != nullptr) {
      u0_ = pmy_pack->phydro->u0;
      w0_ = pmy_pack->phydro->w0;
    } else if (pmy_pack->pmhd != nullptr) {
      u0_ = pmy_pack->pmhd->u0;
      w0_ = pmy_pack->pmhd->w0;
    }
    if (inject_pars.check_temperature || inject_pars.check_beta) {
      gamma_eos = pmy_pack->pmhd->peos->eos_data.gamma;
    }
    if (inject_pars.check_current || inject_pars.check_beta) {
      bcc_ = pmy_pack->pmhd->bcc0;
      auto &bface_ = pmy_pack->pmhd->b0;
      pmy_pack->pmhd->peos->ConsToPrim(u0_,bface_,w0_,bcc_,false,0,(n1-1),0,(n2-1),0,(n3-1));
      if (inject_pars.check_current) {
        const Real thr_val = 1.9999;
        // Allocation based on w0_ allows to reuse indeces
        Kokkos::realloc(
            j_, 
            w0_.extent(0), w0_.extent(1), w0_.extent(2), w0_.extent(3), w0_.extent(4)
        );
        ComputeCurrent(
            inject_pars.current_threshold, thr_val, nmb, coord, bface_, size, indcs,
            j_, curr_max
        );
      }
    }
  }
  auto &injp = inject_pars; // Capture for kernel
  const Real bhspin = coord.bh_spin;
  Kokkos::parallel_reduce("prtcls_injection_checkcondition", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA( const int &idx, int &cell_cnt ) {
        int m = (idx)/nkji;
        int k = (idx - m*nkji)/nji;
        int j = (idx - m*nkji - k*nji)/indcs.nx1;
        int i = (idx - m*nkji - k*nji - j*indcs.nx1)+is;
        k+=ks;
        j+=js;
        // First check geometry constraints
        const Real x1min = size.d_view(m).x1min;
        const Real x1max = size.d_view(m).x1max;
        const Real x2min = size.d_view(m).x2min;
        const Real x2max = size.d_view(m).x2max;
        const Real x3min = size.d_view(m).x3min;
        const Real x3max = size.d_view(m).x3max;
        const Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
        const Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
        const Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
        Real r, th, phi;
        GetBoyerLindquistCoordinates(bhspin, x1v, x2v, x3v, &r, &th, &phi);
        bool use_cell = true;
        use_cell &= ( injp.r_min <= r && r <= injp.r_max );
        use_cell &= ( injp.x3_min <= fabs(x3v) && fabs(x3v) <= injp.x3_max );
        bool th_condition = ( injp.theta_min <= th && th <= injp.theta_max );
        // th_condition |= ( injp.theta_min <= (th+M_PI/2.0) && (th+M_PI/2.0) <= injp.theta_max );
        th_condition |= ( injp.theta_min <= (M_PI-th) && (M_PI-th) <= injp.theta_max );
        use_cell &= ( th_condition );
        use_cell &= ( injp.phi_min <= phi && phi <= injp.phi_max );
        // Check fluid properties if geometric contraints are satisfied
        if (use_cell) {
          if (injp.check_density) { use_cell &= ( w0_(m,IDN,k,j,i) > injp.density_threshold ); }
          if (injp.check_current) { use_cell &= ( j_(m,IDN,k,j,i)  > injp.current_threshold ); }
          /*if (injp.check_asp_ratio) { 
              Real aspect_ratio = ComputeAspectRatio( m, k, j, i, ke, je, ie, 
                  injp.current_threshold, injp.asp_ratio_threshold, bcc_, j_ );
            use_cell = ( use_cell || ( aspect_ratio  > injp.asp_ratio_threshold ) );
          }
          */
          if (injp.check_temperature) { 
            Real pgas = (gamma_eos - 1.0)*w0_(m,IEN,k,j,i); // Assumes ideal EOS
            Real norm_tmprtr = pgas/w0_(m,IDN,k,j,i);
            use_cell &= ( norm_tmprtr > injp.temperature_threshold ); 
          }
          if (injp.check_beta) { 
            // Check whether the value of beta at a given location is significantly 
            // larger than the surrounding
            Real avg_beta = 0.0;
            Real gu[4][4], gl[4][4];
            ComputeMetricAndInverse(x1v,x2v,x3v,coord.is_minkowski,coord.bh_spin,gl,gu); 
            const Real alpha2 = fabs(1.0/gu[0][0]);
            for (int ia = -ng; ia <= ng; ++ia) {
              for (int ib = -ng; ib <= ng; ++ib) {
                for (int ic = -ng; ic <= ng; ++ic) {
                  const int sk = k+ia;
                  const int sj = j+ib;
                  const int si = i+ic;
                  Real aux_pgas = (gamma_eos - 1.0)*w0_(m,IEN,sk,sj,si); // Assumes ideal EOS
                  Real aux_pmag = SQR(bcc_(m,IBX,sk,sj,si))*gl[1][1] + SQR(bcc_(m,IBY,sk,sj,si))*gl[2][2] 
                      + SQR(bcc_(m,IBZ,sk,sj,si))*gl[3][3]
                      + 2.0*bcc_(m,IBX,sk,sj,si)*bcc_(m,IBY,sk,sj,si)*gl[1][2] 
                      + 2.0*bcc_(m,IBX,sk,sj,si)*bcc_(m,IBZ,sk,sj,si)*gl[1][3]
                      + 2.0*bcc_(m,IBY,sk,sj,si)*bcc_(m,IBZ,sk,sj,si)*gl[2][3];
                  aux_pmag *= alpha2;
                  avg_beta += aux_pgas/aux_pmag;
                }
              }
            }
            Real pgas = (gamma_eos - 1.0)*w0_(m,IEN,k,j,i); // Assumes ideal EOS
            Real pmag = SQR(bcc_(m,IBX,k,j,i))*gl[1][1] + SQR(bcc_(m,IBY,k,j,i))*gl[2][2] 
                  + SQR(bcc_(m,IBZ,k,j,i))*gl[3][3]
                  + 2.0*bcc_(m,IBX,k,j,i)*bcc_(m,IBY,k,j,i)*gl[1][2] 
                  + 2.0*bcc_(m,IBX,k,j,i)*bcc_(m,IBZ,k,j,i)*gl[1][3]
                  + 2.0*bcc_(m,IBY,k,j,i)*bcc_(m,IBZ,k,j,i)*gl[2][3];
            pmag *= alpha2;
            avg_beta -= pgas/pmag; // Remove central value to establish "baseline"
            avg_beta /= (SQR(ng)*ng-1);
            use_cell &= ( (pgas/pmag)/avg_beta > injp.beta_threshold ); 
          }
        }
        cells_for_injection(m,k,j,i) = use_cell;
        if (use_cell) { cell_cnt += 1; }
      }, Kokkos::Sum<int>(num_good_cells));
  
  Kokkos::realloc(only_good_cells, num_good_cells, 4);
  Kokkos::View<int> good_cell_idx("good_cell_idx");
  Kokkos::deep_copy(good_cell_idx, 0); // Ensure initialization at 0
  par_for("part_goodcells", DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
    if (cells_for_injection(m,k,j,i)) {
      int cc = Kokkos::atomic_fetch_add(&good_cell_idx(), 1);
      only_good_cells(cc,0) = m;
      only_good_cells(cc,1) = k;
      only_good_cells(cc,2) = j;
      only_good_cells(cc,3) = i;
    }
  });
  return;
}

void Particles::InitializePrtcls(const DvceArray2D<int> &cell_inj, const int &num_good_cells) {
  const int nmb = pmy_pack->nmb_thispack;
  const bool is_gca = is_gca;
  const bool set_radius = inject_pars.init_gyroradius;
  auto &pr = prtcl_rdata;
  auto &pi = prtcl_idata;
  auto &gids = pmy_pack->gids;
  auto &gide = pmy_pack->gide;
  const Real q_over_m = charge_over_mass;
  int &npart = nprtcl_thispack;
  auto &size  = pmy_pack->pmb->mb_size;
  // It will be much more convenient to flip this to check for photons once implemented
  const Real massive = (particle_type == ParticleType::cosmic_ray) ? 1.0 : 0.0;
  auto &coord = pmy_pack->pcoord->coord_data;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int &ng = indcs.ng;
  const int n1 = indcs.nx1 + 2*ng;
  const int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  const int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  const int is = indcs.is; const int ie = indcs.ie;
  const int js = indcs.js; const int je = indcs.je;
  const int ks = indcs.ks; const int ke = indcs.ke;

  const bool flow_align = inject_pars.flow_align; // Capture for kernel
  const Real max_en = inject_pars.energy_max;
  const Real min_en = inject_pars.energy_min;

  Kokkos::Random_XorShift64_Pool<> prtcl_rand(gids);
  DvceArray5D<Real> u0_, w0_;
  DvceArray5D<Real> bcc_;
  if (pmy_pack->phydro != nullptr) {
    u0_ = pmy_pack->phydro->u0;
    w0_ = pmy_pack->phydro->w0;
  } else if (pmy_pack->pmhd != nullptr) {
    u0_ = pmy_pack->pmhd->u0;
    w0_ = pmy_pack->pmhd->w0;
    bcc_ = pmy_pack->pmhd->bcc0;
  }

  par_for("part_init", DevExeSpace(),0,npart-1,
    KOKKOS_LAMBDA(const int p){
      auto prtcl_gen = prtcl_rand.get_state();
      int cc = static_cast<int>(prtcl_gen.frand()*num_good_cells);
      int m = cell_inj(cc,0);
      int k = cell_inj(cc,1);
      int j = cell_inj(cc,2);
      int i = cell_inj(cc,3);

      //Actually initialize the particle
      const Real x1min = size.d_view(m).x1min;
      const Real x1max = size.d_view(m).x1max;
      const Real x2min = size.d_view(m).x2min;
      const Real x2max = size.d_view(m).x2max;
      const Real x3min = size.d_view(m).x3min;
      const Real x3max = size.d_view(m).x3max;
      Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
      Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
      Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
      // Scatter around cell center (still within cell)
      x1v += (0.5 - prtcl_gen.frand())*size.d_view(m).dx1;
      x2v += (0.5 - prtcl_gen.frand())*size.d_view(m).dx2;
      x3v += (0.5 - prtcl_gen.frand())*size.d_view(m).dx3;
      pi(PGID,p) = gids+m;
      pr(IPX,p) = x1v; pr(IPY,p) = x2v; pr(IPZ,p) = x3v;
      Real u[3], b[3];
      b[0] = bcc_(m,IBX,k,j,i);  b[1] = bcc_(m,IBY,k,j,i);  b[2] = bcc_(m,IBZ,k,j,i);
      if (flow_align) { u[0] = w0_(m,IVX,k,j,i); u[1] = w0_(m,IVY,k,j,i); u[2] = w0_(m,IVZ,k,j,i);}
      else { u[0] = 0.1*(0.5 - prtcl_gen.frand()); u[1] = 0.1*(0.5 - prtcl_gen.frand()); u[2] = 0.1*(0.5 - prtcl_gen.frand());}
      Real this_en = min_en + prtcl_gen.frand()*(max_en - min_en);
      prtcl_rand.free_state(prtcl_gen);

      InjectKineticPrtcl( x1v, x2v, x3v, u, b, massive, q_over_m, this_en, max_en, min_en,
                         coord.is_minkowski, coord.bh_spin, set_radius );
      Real gu[4][4], gl[4][4];
      ComputeMetricAndInverse(x1v,x2v,x3v,coord.is_minkowski,coord.bh_spin,gl,gu); 
      Real u_0 = 0.0;
      for (int i1 = 0; i1 < 3; ++i1 ){ 
        for (int i2 = 0; i2 < 3; ++i2 ){
          u_0 += gl[i1+1][i2+1]*u[i1]*u[i2];
        }
      }
      if (!is_gca) {
        pr(IPVX,p) = gl[1][1]*u[0] + gl[1][2]*u[1] + gl[1][3]*u[2];
        pr(IPVY,p) = gl[2][1]*u[0] + gl[2][2]*u[1] + gl[2][3]*u[2];
        pr(IPVZ,p) = gl[3][1]*u[0] + gl[3][2]*u[1] + gl[3][3]*u[2];
      } else {
        pr(IPVX,p) = sqrt(u_0);
        pr(IPVY,p) = 0.001;
      }
  });
  std::cout << "Injected " << npart << " particles in rank " << global_variable::my_rank << "." << std::endl;
  return;
}

} //namespace
