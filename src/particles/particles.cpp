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

  // compute number of particles as real number, since ppc can be < 1
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ncells = indcs.nx1*indcs.nx2*indcs.nx3;
  Real r_npart = ppc*static_cast<Real>((pmy_pack->nmb_thispack)*ncells);
  // then cast to integer
  nprtcl_thispack = static_cast<int>(r_npart);
  average_iteration_number = 0.0;

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
  if (ppush.compare("drift") == 0) {
    pusher = ParticlesPusher::drift;
  } else if (ppush.compare("boris_gr") == 0) {
    charge_over_mass = pin->GetOrAddReal("particles", "charge_over_mass", 1.0);
    pusher = ParticlesPusher::boris_gr;
  } else if (ppush.compare("ham_geo") == 0) {
    max_iter = pin->GetOrAddInteger("particles", "max_iter", 10);
    iter_tolerance = pin->GetOrAddReal("particles", "iter_tolerance", 1.0E-7);
    min_radius = pin->GetOrAddReal("particles", "min_radius", 2.0);
    charge_over_mass = pin->GetOrAddReal("particles", "charge_over_mass", 1.0);
    pusher = ParticlesPusher::ham_geo;
  } else if (ppush.compare("imr") == 0) {
    max_iter = pin->GetOrAddInteger("particles", "max_iter", 10);
    iter_tolerance = pin->GetOrAddReal("particles", "iter_tolerance", 1.0E-7);
    min_radius = pin->GetOrAddReal("particles", "min_radius", 2.0);
    charge_over_mass = pin->GetOrAddReal("particles", "charge_over_mass", 1.0);
    pusher = ParticlesPusher::imr;
  } else if (ppush.compare("gca_gr") == 0) {
		is_gca = true;
    max_iter = pin->GetOrAddInteger("particles", "max_iter", 10);
    iter_tolerance = pin->GetOrAddReal("particles", "iter_tolerance", 1.0E-7);
    min_radius = pin->GetOrAddReal("particles", "min_radius", 2.0);
    charge_over_mass = pin->GetOrAddReal("particles", "charge_over_mass", 1.0);
    pusher = ParticlesPusher::gca_gr;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Particle pusher must be specified in <particles> block" <<std::endl;
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
// ComputeNewdt()
// Find new dt imposed by particle velocities
TaskStatus Particles::NewTimeStep(Driver *pdrive, int stage) {
	auto &pr = prtcl_rdata;
	auto &pi = prtcl_idata;
	const Real spin = pmy_pack->pcoord->coord_data.bh_spin;
	const bool &multi_d = pmy_pack->pmesh->multi_d;
	const bool &three_d = pmy_pack->pmesh->three_d;
  auto gids = pmy_pack->gids;
	const bool is_minkowski = pmy_pack->pcoord->coord_data.is_minkowski;
	auto &mbsize = pmy_pack->pmb->mb_size;
  Real dt1 = std::numeric_limits<float>::max();
  Real dt2 = std::numeric_limits<float>::max();
  Real dt3 = std::numeric_limits<float>::max();

	Kokkos::parallel_reduce("part_newdt",Kokkos::RangePolicy<>(DevExeSpace(),0,(nprtcl_thispack-1)),
		KOKKOS_LAMBDA(const int &p, Real &min_dt1, Real &min_dt2, Real &min_dt3) {
		const Real x[3] = {pr(IPX,p), pr(IPY,p), pr(IPZ,p)};
		const Real u[3] = {pr(IPVX,p), pr(IPVY,p), pr(IPVZ,p)};
    const int m = pi(PGID,p) - gids;
    Real v[3];
    GRRHSPosition(x, u, is_minkowski, spin, v);

    min_dt1 = fmin((mbsize.d_view(m).dx1/fabs(v[0])), min_dt1);
    min_dt2 = fmin((mbsize.d_view(m).dx2/fabs(v[1])), min_dt2);
    min_dt3 = fmin((mbsize.d_view(m).dx3/fabs(v[2])), min_dt3);

  }, Kokkos::Min<Real>(dt1), Kokkos::Min<Real>(dt2),Kokkos::Min<Real>(dt3));
  dtnew = dt1;
  if (pmy_pack->pmesh->multi_d) { dtnew = std::min(dtnew, dt2); }
  if (pmy_pack->pmesh->three_d) { dtnew = std::min(dtnew, dt3); }

  return TaskStatus::complete;
}

} // namespace particles
