//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particles.cpp
//! \brief implementation of Particles class constructor and assorted other functions

#include <iostream>
#include <string>
#include <algorithm>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "bvals/bvals.hpp"
#include "particles.hpp"
#include "hamiltonian_gr.hpp"
#include "coordinates/cartesian_ks.hpp"

namespace particles {
//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

Particles::Particles(MeshBlockPack *ppack, ParameterInput *pin) :
    pmy_pack(ppack) {
  // check this is at least a 2D problem
  if (pmy_pack->pmesh->one_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Particle module only works in 2D/3D" <<std::endl;
    std::exit(EXIT_FAILURE);
  }

  // read number of particles per cell, and calculate number of particles this pack
  Real ppc = pin->GetOrAddReal("particles","ppc",1.0);
  prtcl_cost = pin->GetOrAddReal("particles","prtcl_balance_cost",0.1);
  fail_num = 0;
  average_iteration_number = 0.0;
  max_iteration_number = 0;

  // compute number of particles as real number, since ppc can be < 1
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ncells = indcs.nx1*indcs.nx2*indcs.nx3;
  Real r_npart = ppc*static_cast<Real>((pmy_pack->nmb_thispack)*ncells);
  // then cast to integer
  nprtcl_thispack = static_cast<int>(r_npart);

  // select particle type
  {
    std::string ptype = pin->GetString("particles","particle_type");
    if (ptype.compare("cosmic_ray") == 0) {
      particle_type = ParticleType::cosmic_ray;
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Particle type = '" << ptype << "' not recognized"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // select pusher algorithm
	is_gca = false;
  std::string ppush = pin->GetString("particles","pusher");
  min_radius = pin->GetOrAddReal("particles", "min_radius", 2.0); // Determines when particles have fallen inside the BH
  if (ppush.compare("drift") == 0) {
    pusher = ParticlesPusher::drift;
  } else if (ppush.compare("boris_gr") == 0) {
    charge_over_mass = pin->GetOrAddReal("particles", "charge_over_mass", 1.0);
    pusher = ParticlesPusher::boris_gr;
  } else if (ppush.compare("ham_geo") == 0) {
    max_iter = pin->GetOrAddInteger("particles", "max_iter", 10);
    iter_tolerance = pin->GetOrAddReal("particles", "iter_tolerance", 1.0E-7);
    charge_over_mass = pin->GetOrAddReal("particles", "charge_over_mass", 1.0);
    pusher = ParticlesPusher::ham_geo;
  } else if (ppush.compare("imr") == 0) {
    max_iter = pin->GetOrAddInteger("particles", "max_iter", 10);
    iter_tolerance = pin->GetOrAddReal("particles", "iter_tolerance", 1.0E-7);
    charge_over_mass = pin->GetOrAddReal("particles", "charge_over_mass", 1.0);
    pusher = ParticlesPusher::imr;
  } else if (ppush.compare("gca_gr") == 0) {
		is_gca = true;
    max_iter = pin->GetOrAddInteger("particles", "max_iter", 10);
    iter_tolerance = pin->GetOrAddReal("particles", "iter_tolerance", 1.0E-7);
    charge_over_mass = pin->GetOrAddReal("particles", "charge_over_mass", 1.0);
    pusher = ParticlesPusher::gca_gr;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Particle pusher must be specified in <particles> block" <<std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Initialize struct that gathers all injection parameters after checked that each injection method has the required parameters
  inject_pars.r_min = fmax(min_radius, pin->GetOrAddReal("particles", "r_init_min", 0.0));
  inject_pars.r_max = pin->GetOrAddReal("particles", "r_init_max", 1.0e+8);
  inject_pars.theta_min = pin->GetOrAddReal("particles", "theta_init_min", 0.0);
  inject_pars.theta_max = pin->GetOrAddReal("particles", "theta_init_max", M_PI);
  inject_pars.phi_min = -M_PI; //TODO: Add logic for restricting azimuthal angle at injection
  inject_pars.phi_max = M_PI; //TODO: Add logic for restricting azimuthal angle at injection
  inject_pars.x3_min = pin->GetOrAddReal("particles", "x3_init_min", ppack->pmesh->mesh_size.x3min);
  inject_pars.x3_max = pin->GetOrAddReal("particles", "x3_init_max", ppack->pmesh->mesh_size.x3max);
                                //
  // Calling "GetOrAdd" permanently adds the parameter to the ParameterInput object, and it will be inherited at restart
  // but this is not necessarily desired behvior, hence approach made a bit more verbose
  inject_pars.density_threshold       = 0.0;
  inject_pars.current_threshold       = 0.0;
  inject_pars.asp_ratio_threshold     = 0.0;
  inject_pars.beta_threshold          = 0.0;
  inject_pars.temperature_threshold   = 0.0;
  inject_pars.check_density     = (pin->DoesParameterExist("particles", "density_threshold"));
  if (inject_pars.check_density)  { inject_pars.density_threshold = pin->GetReal("particles", "density_threshold"); }
  inject_pars.check_current     = (pin->DoesParameterExist("particles", "j_threshold"));
  if (inject_pars.check_current)  { inject_pars.current_threshold = pin->GetReal("particles", "j_threshold"); }
  inject_pars.check_asp_ratio   = (pin->DoesParameterExist("particles", "asp_ratio_threshold"));
  if (inject_pars.check_asp_ratio)  { inject_pars.asp_ratio_threshold = pin->GetReal("particles", "asp_ratio_threshold"); }
  inject_pars.check_beta   = (pin->DoesParameterExist("particles", "beta_threshold"));
  if (inject_pars.check_beta)  { inject_pars.beta_threshold = pin->GetReal("particles", "beta_threshold"); }
  inject_pars.check_temperature   = (pin->DoesParameterExist("particles", "temperature_threshold"));
  if (inject_pars.check_temperature)  { inject_pars.temperature_threshold = pin->GetReal("particles", "temperature_threshold"); }

  inject_pars.energy_min = pin->GetOrAddReal("particles", "prtcl_energy_min", 1.0);
  inject_pars.energy_max = pin->GetOrAddReal("particles", "prtcl_energy_max", 1.1);

  inject_pars.try_lim = 25;
  inject_pars.init_gyroradius = pin->GetOrAddBoolean("particles", "set_gyroradius", false);
  
  // select initialization method
  std::string prtcl_init = pin->GetOrAddString("particles","init_method","random");
  if (prtcl_init.compare("random") == 0) {
    inject_pars.flow_align = false;
  } else if (prtcl_init.compare("flow_align") == 0) {
    inject_pars.flow_align = true;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Particle initialization method not recognized." <<std::endl;
    std::exit(EXIT_FAILURE);
  }

  // set dimensions of particle arrays. Note particles only work in 2D/3D
  if (pmy_pack->pmesh->one_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Particles only work in 2D/3D, but 1D problem initialized" <<std::endl;
    std::exit(EXIT_FAILURE);
  }
  switch (particle_type) {
    case ParticleType::cosmic_ray:
      {
        int ndim=4;
        if (pmy_pack->pmesh->three_d) {ndim+=2;}
				if (is_gca) {ndim-=1;} // GCA only has 2 velocity coordinates IPVX is used for parallel component, IPVY for magnetic momentum
        nrdata = ndim;
        nidata = 2;
        break;
      }
    default:
      break;
  }
  Kokkos::realloc(prtcl_rdata, nrdata, nprtcl_thispack);
  Kokkos::realloc(prtcl_idata, nidata, nprtcl_thispack);

  // allocate boundary object
  pbval_part = new ParticlesBoundaryValues(this, pin);
}

//----------------------------------------------------------------------------------------
// destructor

Particles::~Particles() {
  delete pbval_part;
}

//----------------------------------------------------------------------------------------
// CreatePaticleTags()
// Assigns tags to particles (unique integer).  Note that tracked particles are always
// those with tag numbers less than ntrack.

void Particles::CreateParticleTags(ParameterInput *pin) {
  std::string assign = pin->GetOrAddString("particles","assign_tag","index_order");

  // tags are assigned sequentially within this rank, starting at 0 with rank=0
  if (assign.compare("index_order") == 0) {
    int tagstart = 0;
    for (int n=1; n<=global_variable::my_rank; ++n) {
      tagstart += pmy_pack->pmesh->nprtcl_eachrank[n-1];
    }

    auto &pi = prtcl_idata;
    par_for("ptags",DevExeSpace(),0,(nprtcl_thispack-1),
    KOKKOS_LAMBDA(const int p) {
      pi(PTAG,p) = tagstart + p;
    });

  // tags are assigned sequentially across ranks
  } else if (assign.compare("rank_order") == 0) {
    int myrank = global_variable::my_rank;
    int nranks = global_variable::nranks;
    auto &pi = prtcl_idata;
    par_for("ptags",DevExeSpace(),0,(nprtcl_thispack-1),
    KOKKOS_LAMBDA(const int p) {
      pi(PTAG,p) = myrank + nranks*p;
    });

  // tag algorithm not recognized, so quit with error
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Particle tag assinment type = '" << assign << "' not recognized"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

//----------------------------------------------------------------------------------------
// NewTimeStep()
// Find new dt imposed by particle velocities or Larmor frequency.
// Currently this is only compatible with the imr pusher.
TaskStatus Particles::NewTimeStep(Driver *pdrive, int stage) {
	auto &pr = prtcl_rdata;
	auto &pi = prtcl_idata;
	const Real &q_over_m = charge_over_mass;
	auto &b0_ = pmy_pack->pmhd->b0;
	auto &e0_ = pmy_pack->pmhd->efld;
	const Real spin = pmy_pack->pcoord->coord_data.bh_spin;
	const bool &multi_d = pmy_pack->pmesh->multi_d;
	const bool &three_d = pmy_pack->pmesh->three_d;
  auto gids = pmy_pack->gids;
	const bool is_minkowski = pmy_pack->pcoord->coord_data.is_minkowski;
	auto &mbsize = pmy_pack->pmb->mb_size;
	auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int nghst = indcs.ng;
  auto dt = (pmy_pack->pmesh->dt);
  Real dt1 = std::numeric_limits<float>::max();
  Real dt2 = std::numeric_limits<float>::max();
  Real dt3 = std::numeric_limits<float>::max();

	Kokkos::parallel_reduce("part_newdt",Kokkos::RangePolicy<>(DevExeSpace(),0, nprtcl_thispack),
		KOKKOS_LAMBDA(const int &p, Real &min_dt1, Real &min_dt2, Real &min_dt3) {
		const Real x[3] = {pr(IPX,p), pr(IPY,p), pr(IPZ,p)};
		Real u[3] = {pr(IPVX,p), pr(IPVY,p), pr(IPVZ,p)};
    const int m = pi(PGID,p) - gids;
    // Real rhs_v[3];
    Real E[3], B[3];
    // GRRHSVelocity(x, u, is_minkowski, spin, rhs_v);
    bool out_of_bounds = false;
    InterpolateFields( x, b0_, e0_, mbsize, indcs, m, E, B, out_of_bounds );
    // GRLorentz_Terms(x, u, E, B, is_minkowski, spin, q_over_m, rhs_v);
    // u[0] += dt*(rhs_v[0]) ;
    // u[1] += dt*(rhs_v[1]) ;
    // u[2] += dt*(rhs_v[2]) ;

    Real v[3];
    GRRHSPosition(x, u, is_minkowski, spin, v);
    for (int i=0; i<3; ++i) {
      v[i] = std::fabs(v[i]);
    }

    // If a particle is almost at the boundary of a meshblock
    // you have nghost*dx space before the InterpolateFields function goes into segmentation fault
    min_dt1 = 1.0; min_dt2 = 1.0; min_dt3 = 1.0;
    min_dt1 = std::fmin(( mbsize.d_view(m).dx1*nghst/v[0] ), min_dt1);
    min_dt2 = std::fmin(( mbsize.d_view(m).dx2*nghst/v[1] ), min_dt2);
    min_dt3 = std::fmin(( mbsize.d_view(m).dx3*nghst/v[2] ), min_dt3);

    Real glower[4][4], gupper[4][4], ADM[3][3]; // Metric 
    ComputeMetricAndInverse(x[0],x[1],x[2], is_minkowski, spin, glower, gupper); 

    GetUpperAdmMetric( gupper, ADM );
    Real alpha = sqrt(-1.0/gupper[0][0]);
    Real u0 = ADM[0][0]*SQR(u[0]) + ADM[1][1]*SQR(u[1]) + ADM[2][2]*SQR(u[2])
          + 2.0*ADM[0][1]*u[0]*u[1] + 2.0*ADM[0][2]*u[0]*u[2]
          + 2.0*ADM[2][1]*u[2]*u[1];
    u0 = sqrt(u0 + 1.0)/alpha; 

    Real omega = glower[1][1]*SQR(B[0]) + glower[2][2]*SQR(B[1]) + glower[3][3]*SQR(B[2])
            + 2.0*glower[1][2]*B[0]*B[1] + 2.0*glower[1][3]*B[0]*B[2]
            + 2.0*glower[2][3]*B[1]*B[2];
    omega = sqrt( omega );
    omega /= u0;
    omega *= q_over_m;
    min_dt1 = std::fmin(min_dt1, std::fabs(M_PI/omega));

  }, Kokkos::Min<Real>(dt1), Kokkos::Min<Real>(dt2), Kokkos::Min<Real>(dt3));
  dtnew = dt1;
  if (pmy_pack->pmesh->multi_d) { dtnew = std::min(dtnew, dt2); }
  if (pmy_pack->pmesh->three_d) { dtnew = std::min(dtnew, dt3); }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
// CountPartclsPerMB()
// For load_balance, compute how many particles each MB contains.
void Particles::CountPartclsPerMB(int *ppmb) {
	auto &pi = prtcl_idata;
  auto &gids = pmy_pack->gids;
  auto &nmb_total = pmy_pack->pmesh->nmb_total;

  for (int m=0; m<pmy_pack->pmesh->nmb_thisrank; ++m)
    ppmb[m] = 0;

  for (int p=0; p<nprtcl_thispack; ++p) {
    int m = pi(PGID,p) - gids; // Take global MB id, rather than local
    ppmb[m]++;
  }
  return;
}

} // namespace particles
