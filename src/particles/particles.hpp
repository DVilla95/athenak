#ifndef PARTICLES_PARTICLES_HPP_
#define PARTICLES_PARTICLES_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particles.hpp
//  \brief definitions for Particles class

#include <map>
#include <memory>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "bvals/bvals.hpp"
#include "mhd/mhd.hpp"
#include "eos/eos.hpp"

// forward declarations
class MHD;
class EquationOfState;

// constants that enumerate ParticlesPusher options
enum class ParticlesPusher {drift, leap_frog, lagrangian_tracer, lagrangian_mc, boris_gr, ham_geo, gca_gr, imr};

// constants that enumerate ParticleTypes
enum class ParticleType {cosmic_ray};

//----------------------------------------------------------------------------------------
//! \struct ParticlesTaskIDs
//  \brief container to hold TaskIDs of all particles tasks

struct ParticlesTaskIDs {
  TaskID push;
  TaskID newgid;
  TaskID count;
  TaskID irecv;
  TaskID sendp;
  TaskID recvp;
  TaskID csend;
  TaskID crecv;
  TaskID newdt;
};

namespace particles {

struct InjectionParams {
  Real r_min; // Radii for injection
  Real r_max;
  Real theta_min; // Poloidal angles for injection
  Real theta_max;
  Real phi_min; // Azimuthal angles for injection
  Real phi_max;
  Real dens_threshold; // Parameters for injection based on fluid properties
  Real current_threshold;
  Real asp_ratio_threshold;
  Real energy_max; // Parameters for particle energy at injection
  Real energy_min;
  int try_lim;
  bool init_gyroradius; // Initialize based on gyroradius rather than energy
  bool check_asp_ratio; // For current sheet: need to check also aspect ratio other than magnitude
  bool check_current; // For current sheet
  bool check_density; // For density threshold
  bool flow_align; // For velocity init
};

//----------------------------------------------------------------------------------------
//! \class Particles

class Particles {
  friend class ParticlesBoundaryValues;
 public:
  Particles(MeshBlockPack *ppack, ParameterInput *pin);
  ~Particles();

  // data
  ParticleType particle_type;
  int nprtcl_thispack;             // number of particles this MeshBlockPack
  int nrdata, nidata;
//  DvceArray1D<int>  prtcl_gid;     // GID of MeshBlock containing each par
//  DvceArray2D<Real> prtcl_pos;     // positions
//  DvceArray2D<Real> prtcl_vel;     // velocities
  DvceArray2D<Real> prtcl_rdata;   // real number properties each particle (x,v,etc.)
  DvceArray2D<int>  prtcl_idata;   // integer properties each particle (gid, tag, etc.)
  Real dtnew;
  Real iter_tolerance;
  Real average_iteration_number;
  int max_iteration_number;
  int fail_num;
  int max_iter;
  Real min_radius; // This radius is used to destroy particles before they may reach the horizon
  Real charge_over_mass; //Store charge over mass ratio
	bool is_gca; // Store if the system in question has only two velocity components
  Real prtcl_push_safety; // "Safety factor" to ensure time-step doesn't cause issues when iterating
  Real prtcl_cost; // "Cost factor" for load balancing

  ParticlesPusher pusher;
  InjectionParams inject_pars;

  // Boundary communication buffers and functions for particles
  ParticlesBoundaryValues *pbval_part;

  // container to hold names of TaskIDs
  ParticlesTaskIDs id;

  // functions...
  void CreateParticleTags(ParameterInput *pin);
  void AssembleTasks(std::map<std::string, std::shared_ptr<TaskList>> tl);
  void CountPartclsPerMB(int *ppmb);
  TaskStatus Push(Driver *pdriver, int stage);
  TaskStatus NewGID(Driver *pdriver, int stage);
  TaskStatus SendCnt(Driver *pdriver, int stage);
  TaskStatus InitRecv(Driver *pdriver, int stage);
  TaskStatus SendP(Driver *pdriver, int stage);
  TaskStatus RecvP(Driver *pdriver, int stage);
  TaskStatus ClearSend(Driver *pdriver, int stage);
  TaskStatus ClearRecv(Driver *pdriver, int stage);
  TaskStatus NewTimeStep(Driver *pdriver, int stage);
  // void UpdateGIDLB(int &prtclgid, int newrank, int myrank, int mygid, int *pcounter,
  //              DualArray1D<ParticleLocationData> slist, int p);

  void BorisStepGR( const Real dt, const bool only_v );
  void HamiltonianGeodesicsIterations( const Real dt );
  void GRLorentzIterations( const Real dt );
  void GCAIterations( const Real dt );

  // injection/initialization functions
  void SelectCellsForInjection(DvceArray4D<bool> &cell_inj, bool &met_crit);
  void InitializePrtcls(const DvceArray4D<bool> &cell_inj);

 private:
  MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this Particles
};

} // namespace particles
#endif // PARTICLES_PARTICLES_HPP_
