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
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "particles/particles_helpers.hpp"

#include <Kokkos_Random.hpp>

KOKKOS_INLINE_FUNCTION
static bool IsWithinRegion( const int m, const DualArray1D<RegionSize> &size, const RegionIndcs &indcs,
          const Real bhspin, const Kokkos::Random_XorShift64_Pool<> *rand_gen, const int try_limit,
          const Real r_min, const Real r_max, const Real d_min,
          const DvceArray5D<Real> &wprim,
          Real * x1v, Real * x2v, Real * x3v ) {
    auto rand_state = rand_gen->get_state();
    const int is = indcs.is;
    const int js = indcs.js;
    const int ks = indcs.ks;
    const Real x1min = size.d_view(m).x1min;
    const Real x1max = size.d_view(m).x1max;
    const Real x2min = size.d_view(m).x2min;
    const Real x2max = size.d_view(m).x2max;
    const Real x3min = size.d_view(m).x3min;
    const Real x3max = size.d_view(m).x3max;
    Real r, th, phi;
    for (int try_mb = 0; try_mb <= try_limit; ++try_mb) {
      *x1v = x1min + rand_state.frand()*(x1max - x1min);
      *x2v = x2min + rand_state.frand()*(x2max - x2min); 
      *x3v = x3min + rand_state.frand()*(x3max - x3min);
      GetBoyerLindquistCoordinates(bhspin, *x1v, *x2v, *x3v, &r, &th, &phi);
      bool rad_criterium = (r >= r_min && r <= r_max);
      int ip = (*x1v - x1min)/size.d_view(m).dx1 + is;
      int jp = (*x2v - x2min)/size.d_view(m).dx2 + js;
      int kp = (*x3v - x3min)/size.d_view(m).dx3 + ks;
      bool dens_criterium = (wprim(m,IDN,kp,jp,ip) > d_min);
      if ( rad_criterium && dens_criterium) {
        rand_gen->free_state(rand_state);
        return true;
      }
    }
    rand_gen->free_state(rand_state);
    return false;
}

namespace particles {

void Particles::SelectMBsForInjection(DvceArray1D<bool> mb_inj, bool * met_crit) {
  // Reset all MBs to false
  int nmb = pmy_pack->nmb_thispack;
  par_for("reset_mbs_injection", DevExeSpace(), 0, nmb-1,
      KOKKOS_LAMBDA( const int &im ) { mb_inj(im) = false; });

  // Check which MBs satisfy the criterium for injection
  // If none, all particles will be removed from this meshblockpack
  const Real min_rad = min_radius;
  switch (injection_method) {
    case InjectionMethod::random:
      {
        par_for("reset_mbs_injection", DevExeSpace(), 0, nmb-1,
            KOKKOS_LAMBDA( const int &im ) { mb_inj(im) = true; });
        *met_crit = true;
        break;
      }

    case InjectionMethod::density_threshold:
      // If the density in this meshblock pack is everywhere smaller than rho_condition times
      // the rho_min value you're not in the disk
      {
        auto &indcs = pmy_pack->pmesh->mb_indcs;
        const int &ng = indcs.ng;
        const int n1 = indcs.nx1 + 2*ng;
        const int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
        const int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
        const int is = indcs.is;
        const int js = indcs.js;
        const int ks = indcs.ks;
        DvceArray5D<Real> u0_, w0_;
        DvceArray5D<Real> bcc_;
        if (pmy_pack->phydro != nullptr) {
          u0_ = pmy_pack->phydro->u0;
          w0_ = pmy_pack->phydro->w0;
        } else if (pmy_pack->pmhd != nullptr) {
          u0_ = pmy_pack->pmhd->u0;
          w0_ = pmy_pack->pmhd->w0;
          bcc_ = pmy_pack->pmhd->bcc0;
          auto &bface_ = pmy_pack->pmhd->b0;
          pmy_pack->pmhd->peos->ConsToPrim(u0_,bface_,w0_,bcc_,false,0,(n1-1),0,(n2-1),0,(n3-1));
        }
        const Real d_max = crit_max;
        Real dens_max = std::numeric_limits<float>::min();
        const int nmkji = nmb*indcs.nx3*indcs.nx2*indcs.nx1;
        const int nkji = indcs.nx3*indcs.nx2*indcs.nx1;
        const int nji  = indcs.nx2*indcs.nx1;
        Kokkos::parallel_reduce("pgen_mbp_checkcondition", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
        KOKKOS_LAMBDA(const int &idx, Real &max_d) {
          // compute m,k,j,i indices of thread and call function
          int m = (idx)/nkji;
          int k = (idx - m*nkji)/nji;
          int j = (idx - m*nkji - k*nji)/indcs.nx1;
          int i = (idx - m*nkji - k*nji - j*indcs.nx1) + is;
          k += ks;
          j += js;
          mb_inj(m) = ( mb_inj(m) || ( u0_(m,IDN,k,j,i) > d_max ) );
          //Find maximum density in this meshblockpack
          max_d = fmax( u0_(m,IDN,k,j,i), max_d );
        }, Kokkos::Max<Real>(dens_max) );
        *met_crit = ( dens_max > d_max );
        break;
      }

    case InjectionMethod::radius:
      // If the density in this meshblock pack is everywhere smaller than rho_condition times
      // the rho_min value you're not in the disk
      {
        const Real r_max = crit_max;
        const Real r_min = crit_min;
        Real dens_max = std::numeric_limits<float>::min();
        int mbs_in_shell = 0;
        // Initialize particles within a specific spherical shell
        auto &size  = pmy_pack->pmb->mb_size;
        auto &coord = pmy_pack->pcoord->coord_data;
        Real bhspin = coord.bh_spin;
        Kokkos::parallel_reduce("pgen_mbp_checkcondition", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmb),
        KOKKOS_LAMBDA(const int &m, int &mb_count ) {
          // Notice the absolute value
          Real x1min = fabs(size.d_view(m).x1min);
          Real x1max = fabs(size.d_view(m).x1max);
          Real x1i = fmin( x1min, x1max );
          Real x1o = fmax( x1min, x1max );
          Real x2min = fabs(size.d_view(m).x2min);
          Real x2max = fabs(size.d_view(m).x2max);
          Real x2i = fmin( x2min, x2max );
          Real x2o = fmax( x2min, x2max );
          Real x3min = fabs(size.d_view(m).x3min);
          Real x3max = fabs(size.d_view(m).x3max);
          Real x3i = fmin( x3min, x3max );
          Real x3o = fmax( x3min, x3max );
          Real r_i, r_o, th, phi_i, phi_o;
          GetBoyerLindquistCoordinates(bhspin, x1i, x2i, x3i, &r_i, &th, &phi_i);
          GetBoyerLindquistCoordinates(bhspin, x1o, x2o, x3o, &r_o, &th, &phi_o);
          //Determine whether the meshblock with index m has cells within the spherical shell
          mb_inj(m) = ( mb_inj(m) || ( r_o > r_min && r_i < r_max ) );
          if ( mb_inj(m) ) { ++mb_count; }
        }, Kokkos::Sum<int>(mbs_in_shell) );
        
        *met_crit = ( mbs_in_shell > 0 );
        break;
      }
    default:
      break;
  }
}

void Particles::InitializePrtcls(const DvceArray1D<bool> mb_inj) {
  const bool is_gca = is_gca;
  const bool set_radius = init_by_radius;
  auto &pr = prtcl_rdata;
  auto &pi = prtcl_idata;
  auto &gids = pmy_pack->gids;
  auto &gide = pmy_pack->gide;
  const Real min_rad = min_radius;
  const Real q_over_m = charge_over_mass;
  int &npart = nprtcl_thispack;
  auto &size  = pmy_pack->pmb->mb_size;
  const Real max_en = init_max;
  const Real min_en = init_min;
  // It will be much more convenient to flip this to check for photons once implemented
  const Real massive = (particle_type == ParticleType::cosmic_ray) ? 1.0 : 0.0;
  auto &coord = pmy_pack->pcoord->coord_data;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int &ng = indcs.ng;
  const int n1 = indcs.nx1 + 2*ng;
  const int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  const int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  const int is = indcs.is;
  const int js = indcs.js;
  const int ks = indcs.ks;
  DvceArray5D<Real> u0_, w0_;
  DvceArray5D<Real> bcc_;
  if (pmy_pack->phydro != nullptr) {
    u0_ = pmy_pack->phydro->u0;
    w0_ = pmy_pack->phydro->w0;
  } else if (pmy_pack->pmhd != nullptr) {
    u0_ = pmy_pack->pmhd->u0;
    w0_ = pmy_pack->pmhd->w0;
    bcc_ = pmy_pack->pmhd->bcc0;
    auto &bface_ = pmy_pack->pmhd->b0;
    pmy_pack->pmhd->peos->ConsToPrim(u0_,bface_,w0_,bcc_,false,0,(n1-1),0,(n2-1),0,(n3-1));
  }

  const int try_lim = 25;
  const Real bhspin = coord.bh_spin;
  Real r_max = std::numeric_limits<float>::max();
  Real r_min = 0.0;
  Real d_min = 0.0;
  if      (injection_method == InjectionMethod::radius) {r_max = crit_max; r_min = crit_min;}
  else if (injection_method == InjectionMethod::density_threshold) {d_min = crit_min;}

  Kokkos::Random_XorShift64_Pool<> prtcl_rand(gids);

  switch (init_method) {
    case InitMethod::random:
      {
        par_for("part_init", DevExeSpace(),0,npart-1,
          KOKKOS_LAMBDA(const int p){
            int m = 0;
            Real x1v, x2v, x3v;
            bool found_mb = false;
            while(!found_mb){
              auto prtcl_gen = prtcl_rand.get_state();
              m = static_cast<int>(prtcl_gen.frand()*(gide-gids));
              while ( !mb_inj(m) ) {
                m = static_cast<int>(prtcl_gen.frand()*(gide-gids));
              }
              prtcl_rand.free_state(prtcl_gen);
              found_mb = IsWithinRegion( m, size, indcs, bhspin,
                            &prtcl_rand, try_lim,
                            std::fmax(r_min,min_rad), r_max, d_min,
                            w0_,
                            &x1v, &x2v, &x3v );
            } // while(!mb_found)
            //Actually initialize the particle
            pi(PGID,p) = gids+m;
            pr(IPX,p) = x1v; pr(IPY,p) = x2v; pr(IPZ,p) = x3v;
            const Real x1min = size.d_view(m).x1min;
            const Real x2min = size.d_view(m).x2min;
            const Real x3min = size.d_view(m).x3min;
            int ip = (x1v - x1min)/size.d_view(m).dx1 + is;
            int jp = (x2v - x2min)/size.d_view(m).dx2 + js;
            int kp = (x3v - x3min)/size.d_view(m).dx3 + ks;
            Real u[3], b[3];
            b[0] = bcc_(m,IBX,kp,jp,ip);  b[1] = bcc_(m,IBY,kp,jp,ip);  b[2] = bcc_(m,IBZ,kp,jp,ip);
            auto prtcl_gen = prtcl_rand.get_state();
            u[0] = 0.1*(0.5 - prtcl_gen.frand());
            u[1] = 0.1*(0.5 - prtcl_gen.frand());
            u[2] = 0.1*(0.5 - prtcl_gen.frand());
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
        break;
      }
    case InitMethod::flow_align:
      {
        par_for("part_init", DevExeSpace(),0,npart-1,
          KOKKOS_LAMBDA(const int p){
            int m = 0;
            Real x1v, x2v, x3v;
            bool found_mb = false;
            while(!found_mb){
              auto prtcl_gen = prtcl_rand.get_state();
              m = static_cast<int>(prtcl_gen.frand()*(gide-gids));
              while ( !mb_inj[m] ) {
                m = static_cast<int>(prtcl_gen.frand()*(gide-gids));
              }
              prtcl_rand.free_state(prtcl_gen);
              found_mb = IsWithinRegion( m, size, indcs, bhspin,
                            &prtcl_rand, try_lim,
                            std::fmax(r_min,min_rad), r_max, d_min,
                            w0_,
                            &x1v, &x2v, &x3v );
            } // while(!mb_found)
            pi(PGID,p) = gids+m;
            pr(IPX,p) = x1v; pr(IPY,p) = x2v; pr(IPZ,p) = x3v;
            const Real x1min = size.d_view(m).x1min;
            const Real x2min = size.d_view(m).x2min;
            const Real x3min = size.d_view(m).x3min;
            int ip = (x1v - x1min)/size.d_view(m).dx1 + is;
            int jp = (x2v - x2min)/size.d_view(m).dx2 + js;
            int kp = (x3v - x3min)/size.d_view(m).dx3 + ks;                  
            Real u[3], b[3];
            b[0] = bcc_(m,IBX,kp,jp,ip);  b[1] = bcc_(m,IBY,kp,jp,ip);  b[2] = bcc_(m,IBZ,kp,jp,ip);
            u[0] = w0_(m,IVX,kp,jp,ip);   u[1] = w0_(m,IVY,kp,jp,ip);   u[2] = w0_(m,IVZ,kp,jp,ip);
            auto prtcl_gen = prtcl_rand.get_state();
            if ( fabs(u[0]*u[1]*u[2]) < 1.0E-10 ) { // If fluid velocity is too small, matching this_en can take too long, reset to random
              u[0] = 0.1*(0.5 - prtcl_gen.frand());
              u[1] = 0.1*(0.5 - prtcl_gen.frand());
              u[2] = 0.1*(0.5 - prtcl_gen.frand());
            }
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
        break;
      }
    default:
      break;
  }
  std::cout << "Injected " << npart << " particles in rank " << global_variable::my_rank << "." << std::endl;
}

} //namespace
