//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hamiltonian_gr.hpp
//!  \brief
//!  Contains functions used by multiple pushers to implement the hamiltonian scheme of Bacchini et al. 2018 (doi.org/10.3847/1538-4365/aac9ca

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "particles.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/cartesian_ks.hpp"

//----------------------------------------------------------------------------------------
//! \fn  void GetUpperAdmMetric
//  \brief
KOKKOS_INLINE_FUNCTION
void GetUpperAdmMetric( const Real inputMat[][4], Real outputMat[][3] ){
	for (int i1 = 0; i1 < 3; ++i1 ){ 
		for (int i2 = 0; i2 < 3; ++i2 ){ 
		outputMat[i1][i2] = inputMat[i1+1][i2+1] - inputMat[0][i2+1]*inputMat[i1+1][0]/inputMat[0][0];
		}
	}
}

//----------------------------------------------------------------------------------------
//! \fn  void SingleTermHelper_Position
//! \brief
//! Helper function to reduce amount of repetition in HamiltonEquation_Position
//! Notice that beta has opposite sign to "correct" one
KOKKOS_INLINE_FUNCTION
void SingleTermHelper_Position(const Real * u_0, const Real * u_1, const Real alpha, const Real beta, const Real ADM[][3], const int dir, Real * H){

	Real massive = 1.0; //todo for photons/massless particles this needs to be 0: condition on ptype
	Real U_1, U_0, perp;
	int i1, i2;
	if (dir == 0) { i1 = 1; i2 = 2; }
	else if (dir == 1) { i1 = 0; i2 = 2; }
	else if (dir == 2) { i1 = 0; i2 = 1; }

	perp = ADM[i1][i1]*SQR(u_0[i1]) + ADM[i2][i2]*SQR(u_0[i2]) + 2.0*ADM[i1][i2]*u_0[i1]*u_0[i2];
	U_1 = ADM[dir][dir]*SQR(u_1[dir]) + 2.0*ADM[dir][i1]*u_1[dir]*u_1[i1] + 2.0*ADM[dir][i2]*u_1[dir]*u_1[i2];
	U_1 += perp;	
	U_1 = sqrt(U_1 + massive); 
	U_0 = ADM[dir][dir]*SQR(u_0[dir]) + 2.0*ADM[dir][i1]*u_0[dir]*u_0[i1] + 2.0*ADM[dir][i2]*u_0[dir]*u_0[i2];
	U_0 += perp;	
	U_0 = sqrt(U_0 + massive); 
	// Alpha term
	*H += alpha*(
		ADM[dir][dir]*(u_1[dir]+u_0[dir]) + 2.0*ADM[dir][i1]*u_1[i1] + 2.0*ADM[dir][i2]*u_1[i2]
	       	)/(U_0 + U_1);
	// Beta term
	*H += beta;
}

//----------------------------------------------------------------------------------------
//! \fn  void HamiltonEquation_Position
//  \brief
// Following function largely implented based on the appendix in Bacchini et al. 2018 (doi.org/10.3847/1538-4365/aac9ca
// It's an expression of the derivative of the ADM Hamiltonian for geodesics with respect to the velocity
// That prevents singularities for small increments and is used to compute the time derivative of the position
// Would be nice to come up with a way to compute these terms algorithmically, rather than by hard coding, but thus far I wasn't able to
KOKKOS_INLINE_FUNCTION
void HamiltonEquation_Position(const Real * x_0, const Real * x_1, const Real * u_0, const Real * u_1, const Real spin, Real * H){

	const bool is_minkowski = false; //Since this function is only for the GR pusher, this can be kept as a ``constant''
	Real glower[4][4], gupper[4][4], ADM_g[3][3]; // Metric components
	Real aux_u0[3], aux_u1[3]; //Send these to helper function on a 'per-case' basis
				   // Assembling these requires referring to the paper for the correct combinations

	for (int i=0; i<3; ++i){ H[i] = 0.0; }

	//Metric with all old positions
	//Common to all terms
	ComputeMetricAndInverse(x_0[0],x_0[1],x_0[2], is_minkowski, spin, glower, gupper); 
	GetUpperAdmMetric( gupper, ADM_g );
	
	aux_u0[0] = u_0[0]; aux_u0[1] = u_0[1]; aux_u0[2] = u_0[2];
	aux_u1[0] = u_1[0]; aux_u1[1] = u_0[1]; aux_u1[2] = u_0[2];
	SingleTermHelper_Position(aux_u0, aux_u1, sqrt(-1.0/gupper[0][0]), gupper[0][1]/gupper[0][0], ADM_g, 0, &H[0]);

	aux_u1[0] = u_0[0]; aux_u1[1] = u_1[1]; aux_u1[2] = u_0[2];
	SingleTermHelper_Position(aux_u0, aux_u1, sqrt(-1.0/gupper[0][0]), gupper[0][2]/gupper[0][0], ADM_g, 1, &H[1]);

	aux_u1[0] = u_0[0]; aux_u1[1] = u_0[1]; aux_u1[2] = u_1[2];
	SingleTermHelper_Position(aux_u0, aux_u1, sqrt(-1.0/gupper[0][0]), gupper[0][3]/gupper[0][0], ADM_g, 2, &H[2]);
	
	//Metric with all new positions
	//Common to all terms
	ComputeMetricAndInverse(x_1[0],x_1[1],x_1[2], is_minkowski, spin, glower, gupper); 
	GetUpperAdmMetric( gupper, ADM_g );

	aux_u0[0] = u_0[0]; aux_u0[1] = u_1[1]; aux_u0[2] = u_1[2];
	aux_u1[0] = u_1[0]; aux_u1[1] = u_1[1]; aux_u1[2] = u_1[2];
	SingleTermHelper_Position(aux_u0, aux_u1, sqrt(-1.0/gupper[0][0]), gupper[0][1]/gupper[0][0], ADM_g, 0, &H[0]);

	aux_u0[0] = u_1[0]; aux_u0[1] = u_0[1]; aux_u0[2] = u_1[2];
	SingleTermHelper_Position(aux_u0, aux_u1, sqrt(-1.0/gupper[0][0]), gupper[0][2]/gupper[0][0], ADM_g, 1, &H[1]);

	aux_u0[0] = u_1[0]; aux_u0[1] = u_1[1]; aux_u0[2] = u_0[2];
	SingleTermHelper_Position(aux_u0, aux_u1, sqrt(-1.0/gupper[0][0]), gupper[0][3]/gupper[0][0], ADM_g, 2, &H[2]);

	//
	//Metric with x_0[0], x_1[1], x_1[2]
	//Common to terms 0 and 2
	ComputeMetricAndInverse(x_0[0],x_1[1],x_1[2], is_minkowski, spin, glower, gupper); 
	GetUpperAdmMetric( gupper, ADM_g );

	aux_u0[0] = u_0[0]; aux_u0[1] = u_1[1]; aux_u0[2] = u_1[2];
	SingleTermHelper_Position(aux_u0, aux_u1, sqrt(-1.0/gupper[0][0]), gupper[0][1]/gupper[0][0], ADM_g, 0, &H[0]);

	aux_u0[0] = u_0[0]; aux_u0[1] = u_1[1]; aux_u0[2] = u_0[2];
	aux_u1[0] = u_0[0]; aux_u1[1] = u_1[1]; aux_u1[2] = u_1[2];
	SingleTermHelper_Position(aux_u0, aux_u1, sqrt(-1.0/gupper[0][0]), gupper[0][3]/gupper[0][0], ADM_g, 2, &H[2]);

	//
	//Metric with x_1[0], x_0[1], x_0[2]
	//Common to terms 0 and 1
	ComputeMetricAndInverse(x_1[0],x_0[1],x_0[2], is_minkowski, spin, glower, gupper); 
	GetUpperAdmMetric( gupper, ADM_g );

	aux_u0[0] = u_0[0]; aux_u0[1] = u_0[1]; aux_u0[2] = u_0[2];
	aux_u1[0] = u_1[0]; aux_u1[1] = u_0[1]; aux_u1[2] = u_0[2];
	SingleTermHelper_Position(aux_u0, aux_u1, sqrt(-1.0/gupper[0][0]), gupper[0][1]/gupper[0][0], ADM_g, 0, &H[0]);

	aux_u0[0] = u_1[0]; aux_u0[1] = u_0[1]; aux_u0[2] = u_0[2];
	aux_u1[0] = u_1[0]; aux_u1[1] = u_1[1]; aux_u1[2] = u_0[2];
	SingleTermHelper_Position(aux_u0, aux_u1, sqrt(-1.0/gupper[0][0]), gupper[0][2]/gupper[0][0], ADM_g, 1, &H[1]);

	//
	//Metric with x_1[0], x_0[1], x_1[2]
	//Common to terms 0 and 1
	ComputeMetricAndInverse(x_1[0],x_0[1],x_1[2], is_minkowski, spin, glower, gupper); 
	GetUpperAdmMetric( gupper, ADM_g );

	aux_u0[0] = u_0[0]; aux_u0[1] = u_0[1]; aux_u0[2] = u_1[2];
	aux_u1[0] = u_1[0]; aux_u1[1] = u_0[1]; aux_u1[2] = u_1[2];
	SingleTermHelper_Position(aux_u0, aux_u1, sqrt(-1.0/gupper[0][0]), gupper[0][1]/gupper[0][0], ADM_g, 0, &H[0]);

	aux_u0[0] = u_1[0]; aux_u0[1] = u_0[1]; aux_u0[2] = u_1[2];
	aux_u1[0] = u_1[0]; aux_u1[1] = u_1[1]; aux_u1[2] = u_1[2];
	SingleTermHelper_Position(aux_u0, aux_u1, sqrt(-1.0/gupper[0][0]), gupper[0][2]/gupper[0][0], ADM_g, 1, &H[1]);

	//
	//Metric with x_0[0], x_0[1], x_1[2]
	//Common to terms 0 and 2
	ComputeMetricAndInverse(x_0[0],x_0[1],x_1[2], is_minkowski, spin, glower, gupper); 
	GetUpperAdmMetric( gupper, ADM_g );

	aux_u0[0] = u_0[0]; aux_u0[1] = u_0[1]; aux_u0[2] = u_1[2];
	aux_u1[0] = u_1[0]; aux_u1[1] = u_0[1]; aux_u1[2] = u_1[2];
	SingleTermHelper_Position(aux_u0, aux_u1, sqrt(-1.0/gupper[0][0]), gupper[0][1]/gupper[0][0], ADM_g, 0, &H[0]);

	aux_u0[0] = u_0[0]; aux_u0[1] = u_0[1]; aux_u0[2] = u_0[2];
	aux_u1[0] = u_0[0]; aux_u1[1] = u_0[1]; aux_u1[2] = u_1[2];
	SingleTermHelper_Position(aux_u0, aux_u1, sqrt(-1.0/gupper[0][0]), gupper[0][3]/gupper[0][0], ADM_g, 2, &H[2]);

	//
	//Metric with x_1[0], x_1[1], x_0[2]
	//Common to terms 1 and 2
	ComputeMetricAndInverse(x_1[0],x_1[1],x_0[2], is_minkowski, spin, glower, gupper); 
	GetUpperAdmMetric( gupper, ADM_g );

	aux_u0[0] = u_1[0]; aux_u0[1] = u_0[1]; aux_u0[2] = u_0[2];
	aux_u1[0] = u_1[0]; aux_u1[1] = u_1[1]; aux_u1[2] = u_0[2];
	SingleTermHelper_Position(aux_u0, aux_u1, sqrt(-1.0/gupper[0][0]), gupper[0][2]/gupper[0][0], ADM_g, 1, &H[1]);

	aux_u0[0] = u_1[0]; aux_u0[1] = u_1[1]; aux_u0[2] = u_0[2];
	aux_u1[0] = u_1[0]; aux_u1[1] = u_1[1]; aux_u1[2] = u_1[2];
	SingleTermHelper_Position(aux_u0, aux_u1, sqrt(-1.0/gupper[0][0]), gupper[0][3]/gupper[0][0], ADM_g, 2, &H[2]);

	//
	//Metric with x_0[0], x_1[1], x_0[2]
	//Common to terms 1 and 2
	ComputeMetricAndInverse(x_0[0],x_1[1],x_0[2], is_minkowski, spin, glower, gupper); 
	GetUpperAdmMetric( gupper, ADM_g );

	aux_u0[0] = u_0[0]; aux_u0[1] = u_0[1]; aux_u0[2] = u_0[2];
	aux_u1[0] = u_0[0]; aux_u1[1] = u_1[1]; aux_u1[2] = u_0[2];
	SingleTermHelper_Position(aux_u0, aux_u1, sqrt(-1.0/gupper[0][0]), gupper[0][2]/gupper[0][0], ADM_g, 1, &H[1]);

	aux_u0[0] = u_0[0]; aux_u0[1] = u_1[1]; aux_u0[2] = u_0[2];
	aux_u1[0] = u_0[0]; aux_u1[1] = u_1[1]; aux_u1[2] = u_1[2];
	SingleTermHelper_Position(aux_u0, aux_u1, sqrt(-1.0/gupper[0][0]), gupper[0][3]/gupper[0][0], ADM_g, 2, &H[2]);
	
	for (int i=0; i<3; ++i){ H[i] /= 6.0; }
}
//----------------------------------------------------------------------------------------
//! \fn  void ComputeAndAddSingleTerm_Velocity
//  \brief
KOKKOS_INLINE_FUNCTION
void ComputeAndAddSingleTerm_Velocity(const Real gu_0[4][4], const Real gu_1[4][4], const Real * u, Real * H){
	Real U_0, U_1, aux, beta_t;
	Real ADM_0[3][3], ADM_1[3][3];
	Real massive = 1.0; //TODO for photons/massless particles this needs to be 0: condition on ptype

	GetUpperAdmMetric( gu_0, ADM_0 );
	GetUpperAdmMetric( gu_1, ADM_1 );

	U_0 = 0.0;
	for (int i1 = 0; i1 < 3; ++i1 ){
					for (int i2 = 0; i2 < 3; ++i2 ){
					U_0 += ADM_0[i1][i2]*u[i1]*u[i2];
					}
	}
	U_0 = sqrt(U_0 + massive);
	U_1 = 0.0;
	for (int i1 = 0; i1 < 3; ++i1 ){
					for (int i2 = 0; i2 < 3; ++i2 ){
					U_1 += ADM_1[i1][i2]*u[i1]*u[i2];
					}
	}
	U_1 = sqrt(U_1 + massive);
	aux = 0.0;
	for (int i1 = 0; i1 < 3; ++i1 ){
					for (int i2 = 0; i2 < 3; ++i2 ){
					aux += (ADM_1[i1][i2] - ADM_0[i1][i2])*u[i1]*u[i2];
					}
	}
	beta_t = 0.0;
	for (int i1 = 0; i1 < 3; ++i1 ){
					beta_t += ( -gu_1[0][i1+1]/gu_1[0][0] + gu_0[0][i1+1]/gu_0[0][0])*u[i1];
	}

	*H -= 0.5*(U_0 + U_1)*( sqrt(-1.0/gu_1[0][0]) - sqrt(-1.0/gu_0[0][0]) ) ;
	*H -= 0.5*aux*( sqrt(-1.0/gu_1[0][0]) + sqrt(-1.0/gu_0[0][0]) )/(U_1 + U_0) ;
	*H += beta_t ;
}

//----------------------------------------------------------------------------------------
//! \fn  void ComputeAndAddSingleTerm_Velocity
//  \brief      
// Overload previous function to use the derivative of the metric if needed
KOKKOS_INLINE_FUNCTION
void ComputeAndAddSingleTerm_Velocity(const Real gu_0[4][4], const Real gu_1[4][4], const Real g_der[4][4], const Real * u, Real * H){
	Real U_0, U_1, aux, beta_t;
	Real ADM_0[3][3], ADM_1[3][3], ADM_der[3][3];
	Real massive = 1.0; //TODO for photons/massless particles this needs to be 0: condition on ptype

	GetUpperAdmMetric( gu_0, ADM_0 );
	GetUpperAdmMetric( gu_1, ADM_1 );
	for (int i1 = 0; i1 < 3; ++i1 ){
					for (int i2 = 0; i2 < 3; ++i2 ){ 
					ADM_der[i1][i2] = g_der[i1+1][i2+1]
									+ g_der[0][0]*gu_0[0][i1+1]*gu_0[0][i2+1]/SQR(gu_0[0][0])
									- 2.0*g_der[0][i1+1]*gu_0[0][i2+1]/gu_0[0][0];
					}
	}
	U_0 = 0.0;
	for (int i1 = 0; i1 < 3; ++i1 ){ 
					for (int i2 = 0; i2 < 3; ++i2 ){
					U_0 += ADM_0[i1][i2]*u[i1]*u[i2];
					}
	}       
	U_0 = sqrt(U_0 + massive); 
	U_1 = 0.0;      
	for (int i1 = 0; i1 < 3; ++i1 ){
					for (int i2 = 0; i2 < 3; ++i2 ){
					U_1 += ADM_1[i1][i2]*u[i1]*u[i2];
					}
	}       
	U_1 = sqrt(U_1 + massive); 
	aux = 0.0;
	for (int i1 = 0; i1 < 3; ++i1 ){
					for (int i2 = 0; i2 < 3; ++i2 ){
					aux += ADM_der[i1][i2]*u[i1]*u[i2];
					}
	}       
	beta_t = 0.0;
	for (int i1 = 0; i1 < 3; ++i1 ){
					beta_t -= (g_der[0][i1+1] - gu_0[0][i1+1]*g_der[0][0]/gu_0[0][0])/gu_0[0][0]*u[i1];
	}
	
	*H -= 0.5*(U_0 + U_1)*( 0.5*sqrt(-1.0/gu_0[0][0])*g_der[0][0]/fabs(gu_0[0][0]) ) ;
	*H -= 0.5*aux*( sqrt(-1.0/gu_1[0][0]) + sqrt(-1.0/gu_0[0][0]) )/(U_1 + U_0) ;
	*H += beta_t ;
}       

//----------------------------------------------------------------------------------------
//! \fn  void SingleTermHelper_Velocity
//! \brief
//! Helper function to reduce amount of repetition in HamiltonEquation_Velocity
KOKKOS_INLINE_FUNCTION
void SingleTermHelper_Velocity(const Real * x_0, const Real * x_1, const Real * u, const bool use_der, const int dir, const Real x_step, const Real spin, Real * H){

	const bool is_minkowski = false; //Since this function is only for the GR pusher, this can be kept as a ``constant''
	Real gl[4][4], gu_0[4][4], gu_1[4][4]; // Metric components
	Real aux_gu_0[4][4], aux_gu_1[4][4]; // Metric components
	int dir_fac[3] = {0,0,0}; 
	dir_fac[dir] = 1; // Use this to perform step only in direction of interest

	ComputeMetricAndInverse(x_0[0],x_0[1],x_0[2], is_minkowski, spin, gl, gu_0); 
	ComputeMetricAndInverse(x_1[0],x_1[1],x_1[2], is_minkowski, spin, gl, gu_1); 

	if (use_der){
		ComputeMetricAndInverse(x_0[0]-x_step*dir_fac[0], x_0[1]-x_step*dir_fac[1], x_0[2]-x_step*dir_fac[2],
			       	is_minkowski, spin, gl, aux_gu_1); 
		ComputeMetricAndInverse(x_0[0]+x_step*dir_fac[0], x_0[1]+x_step*dir_fac[1], x_0[2]+x_step*dir_fac[2],
			       	is_minkowski, spin, gl, aux_gu_0); 
	        for(int i=0; i<4; ++i){
			for(int j=0; j<4; ++j){
				aux_gu_0[i][j] -= aux_gu_1[i][j];
				aux_gu_0[i][j] /= (2.0*x_step);
			}
		}	
		ComputeAndAddSingleTerm_Velocity(gu_0, gu_1, aux_gu_0, u, H);
	}else{
		ComputeAndAddSingleTerm_Velocity(gu_0, gu_1, u, H);
	}
									     //
}

//----------------------------------------------------------------------------------------
//! \fn  void HamiltonEquation_Velocity
//!  \brief
//! Following function largely implented based on the appendix in Bacchini et al. 2018 (doi.org/10.3847/1538-4365/aac9ca
//! It's an expression of the derivative of the ADM Hamiltonian for geodesics with respect to the position
//! That prevents singularities for small increments and is used to compute the time derivative of the velocity
//! Would be nice to come up with a way to compute these terms algorithmically, rather than by hard coding, but thus far I wasn't able to
KOKKOS_INLINE_FUNCTION
void HamiltonEquation_Velocity(const Real * x_0, const Real * x_1, const Real * u_0, const Real * u_1, const Real x_step, const Real spin, const Real it_tol, Real * H){

	Real incr[3], u[3];
       	// Difference between old and new points might be too small
	// Use drivative instead
	bool use_derivative[3]; // array to determine whether or not the derivative is needed
	Real aux_x0[3], aux_x1[3]; // Send these to helper function, but assemble them case by case following energy conserving scheme

	for (int i=0; i<3; ++i){ H[i] = 0.0; }
	incr[0] = x_1[0] - x_0[0];
	use_derivative[0] = (fabs(incr[0]) < sqrt(it_tol)) ? true : false;
	incr[1] = x_1[1] - x_0[1];
	use_derivative[1] = (fabs(incr[1]) < sqrt(it_tol)) ? true : false;
	incr[2] = x_1[2] - x_0[2];
	use_derivative[2] = (fabs(incr[2]) < sqrt(it_tol)) ? true : false;

	//Terms with all old velocities
	//Common to all directions
	u[0] = u_0[0]; u[1] = u_0[1]; u[2] = u_0[2]; 
	aux_x0[0] = x_0[0]; aux_x0[1] = x_0[1]; aux_x0[2] = x_0[2];
	aux_x1[0] = x_1[0]; aux_x1[1] = x_0[1]; aux_x1[2] = x_0[2];
	SingleTermHelper_Velocity(aux_x0, aux_x1, u, use_derivative[0], 0, x_step, spin, &H[0]);

	aux_x1[0] = x_0[0]; aux_x1[1] = x_1[1]; aux_x1[2] = x_0[2];
	SingleTermHelper_Velocity(aux_x0, aux_x1, u, use_derivative[1], 1, x_step, spin, &H[1]);
	
	aux_x1[0] = x_0[0]; aux_x1[1] = x_0[1]; aux_x1[2] = x_1[2];
	SingleTermHelper_Velocity(aux_x0, aux_x1, u, use_derivative[2], 2, x_step, spin, &H[2]);

	//Terms with all new velocities
	//Common to all directions
	u[0] = u_1[0]; u[1] = u_1[1]; u[2] = u_1[2]; 
	aux_x0[0] = x_0[0]; aux_x0[1] = x_1[1]; aux_x0[2] = x_1[2];
	aux_x1[0] = x_1[0]; aux_x1[1] = x_1[1]; aux_x1[2] = x_1[2];
	SingleTermHelper_Velocity(aux_x0, aux_x1, u, use_derivative[0], 0, x_step, spin, &H[0]);

	aux_x0[0] = x_1[0]; aux_x0[1] = x_0[1]; aux_x0[2] = x_1[2];
	SingleTermHelper_Velocity(aux_x0, aux_x1, u, use_derivative[1], 1, x_step, spin, &H[1]);

	aux_x0[0] = x_1[0]; aux_x0[1] = x_1[1]; aux_x0[2] = x_0[2];
	SingleTermHelper_Velocity(aux_x0, aux_x1, u, use_derivative[2], 2, x_step, spin, &H[2]);

	//Terms with new velocities for y and z
	//Common to terms 0 and 2
	u[0] = u_0[0]; u[1] = u_1[1]; u[2] = u_1[2]; 
	aux_x0[0] = x_0[0]; aux_x0[1] = x_1[1]; aux_x0[2] = x_1[2];
	aux_x1[0] = x_1[0]; aux_x1[1] = x_1[1]; aux_x1[2] = x_1[2];
	SingleTermHelper_Velocity(aux_x0, aux_x1, u, use_derivative[0], 0, x_step, spin, &H[0]);

	aux_x0[0] = x_0[0]; aux_x0[1] = x_1[1]; aux_x0[2] = x_0[2];
	aux_x1[0] = x_0[0]; aux_x1[1] = x_1[1]; aux_x1[2] = x_1[2];
	SingleTermHelper_Velocity(aux_x0, aux_x1, u, use_derivative[2], 2, x_step, spin, &H[2]);
	
	//Terms with new velocities for x and y
	//Common to terms 1 and 2
	u[0] = u_1[0]; u[1] = u_1[1]; u[2] = u_0[2]; 
	aux_x0[0] = x_1[0]; aux_x0[1] = x_0[1]; aux_x0[2] = x_0[2];
	aux_x1[0] = x_1[0]; aux_x1[1] = x_1[1]; aux_x1[2] = x_0[2];
	SingleTermHelper_Velocity(aux_x0, aux_x1, u, use_derivative[1], 1, x_step, spin, &H[1]);
	
	aux_x0[0] = x_1[0]; aux_x0[1] = x_1[1]; aux_x0[2] = x_0[2];
	aux_x1[0] = x_1[0]; aux_x1[1] = x_1[1]; aux_x1[2] = x_1[2];
	SingleTermHelper_Velocity(aux_x0, aux_x1, u, use_derivative[2], 2, x_step, spin, &H[2]);

	//Terms with new velocities for x and z
	//Common to terms 0 and 1
	u[0] = u_1[0]; u[1] = u_0[1]; u[2] = u_1[2]; 
	aux_x0[0] = x_0[0]; aux_x0[1] = x_0[1]; aux_x0[2] = x_1[2];
	aux_x1[0] = x_1[0]; aux_x1[1] = x_0[1]; aux_x1[2] = x_1[2];
	SingleTermHelper_Velocity(aux_x0, aux_x1, u, use_derivative[0], 0, x_step, spin, &H[0]);
	
	aux_x0[0] = x_1[0]; aux_x0[1] = x_0[1]; aux_x0[2] = x_1[2];
	aux_x1[0] = x_1[0]; aux_x1[1] = x_1[1]; aux_x1[2] = x_1[2];
	SingleTermHelper_Velocity(aux_x0, aux_x1, u, use_derivative[1], 1, x_step, spin, &H[1]);

	//Terms with new velocities for y only
	//Common to terms 1 and 2
	u[0] = u_0[0]; u[1] = u_1[1]; u[2] = u_0[2]; 
	aux_x0[0] = x_0[0]; aux_x0[1] = x_0[1]; aux_x0[2] = x_0[2];
	aux_x1[0] = x_0[0]; aux_x1[1] = x_1[1]; aux_x1[2] = x_0[2];
	SingleTermHelper_Velocity(aux_x0, aux_x1, u, use_derivative[1], 1, x_step, spin, &H[1]);
	
	aux_x0[0] = x_0[0]; aux_x0[1] = x_1[1]; aux_x0[2] = x_0[2];
	aux_x1[0] = x_0[0]; aux_x1[1] = x_1[1]; aux_x1[2] = x_1[2];
	SingleTermHelper_Velocity(aux_x0, aux_x1, u, use_derivative[2], 2, x_step, spin, &H[2]);

	//Terms with new velocities for z only
	//Common to terms 0 and 2
	u[0] = u_0[0]; u[1] = u_0[1]; u[2] = u_1[2]; 
	aux_x0[0] = x_0[0]; aux_x0[1] = x_0[1]; aux_x0[2] = x_1[2];
	aux_x1[0] = x_1[0]; aux_x1[1] = x_0[1]; aux_x1[2] = x_1[2];
	SingleTermHelper_Velocity(aux_x0, aux_x1, u, use_derivative[0], 0, x_step, spin, &H[0]);
	
	aux_x0[0] = x_0[0]; aux_x0[1] = x_0[1]; aux_x0[2] = x_0[2];
	aux_x1[0] = x_0[0]; aux_x1[1] = x_0[1]; aux_x1[2] = x_1[2];
	SingleTermHelper_Velocity(aux_x0, aux_x1, u, use_derivative[2], 2, x_step, spin, &H[2]);

	//Terms with new velocities for x only
	//Common to terms 0 and 1
	u[0] = u_1[0]; u[1] = u_0[1]; u[2] = u_0[2]; 
	aux_x0[0] = x_0[0]; aux_x0[1] = x_0[1]; aux_x0[2] = x_0[2];
	aux_x1[0] = x_1[0]; aux_x1[1] = x_0[1]; aux_x1[2] = x_0[2];
	SingleTermHelper_Velocity(aux_x0, aux_x1, u, use_derivative[0], 0, x_step, spin, &H[0]);

	aux_x0[0] = x_1[0]; aux_x0[1] = x_0[1]; aux_x0[2] = x_0[2];
	aux_x1[0] = x_1[0]; aux_x1[1] = x_1[1]; aux_x1[2] = x_0[2];
	SingleTermHelper_Velocity(aux_x0, aux_x1, u, use_derivative[1], 1, x_step, spin, &H[1]);

	for (int i=0; i<3; ++i){
	       	H[i] /= 6.0;
		// When using the derivative the division by the increment is done to compute the derivatives
		if (!use_derivative[i]) { H[i] /= incr[i]; }

       	}
}

//----------------------------------------------------------------------------------------
//! \fn  void ComputeDeterminant3
//  \brief Compute the determinant of a 3x3 matrix
KOKKOS_INLINE_FUNCTION
void ComputeDeterminant3( const Real inputMat[][3], Real &determinant ){

	determinant = inputMat[0][0]*inputMat[1][1]*inputMat[2][2] + inputMat[0][1]*inputMat[1][2]*inputMat[2][0]
		+ inputMat[1][0]*inputMat[2][1]*inputMat[0][2]
		- inputMat[2][0]*inputMat[1][1]*inputMat[0][2] - inputMat[1][0]*inputMat[0][1]*inputMat[2][2]
		- inputMat[2][1]*inputMat[1][2]*inputMat[1][1];

}

//----------------------------------------------------------------------------------------
//! \fn  void ComputeInverseMatrix3
//  \brief Compute the inverse of a 3x3 matrix
KOKKOS_INLINE_FUNCTION
void ComputeInverseMatrix3( const Real inputMat[][3], Real outputMat[][3] ){

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
//  \brief Interpolate cell field to particle location,
KOKKOS_INLINE_FUNCTION
void InterpolateFields( const Real * prtcl_x, const DvceFaceFld4D<Real> &b0_, const DvceEdgeFld4D<Real> &e0_,
				const DualArray1D<RegionSize> &mbsize, const RegionIndcs &indcs, const int m,
			  Real * E, Real * B ){

	int ip = (prtcl_x[0] - mbsize.d_view(m).x1min)/mbsize.d_view(m).dx1 + indcs.is;
	int jp = (prtcl_x[1] - mbsize.d_view(m).x2min)/mbsize.d_view(m).dx2 + indcs.js;
	int kp = (prtcl_x[2] - mbsize.d_view(m).x3min)/mbsize.d_view(m).dx3 + indcs.ks;
	Real &x1min = mbsize.d_view(m).x1min;
	Real &x2min = mbsize.d_view(m).x2min;
	Real &x3min = mbsize.d_view(m).x3min;
	Real &x1max = mbsize.d_view(m).x1max;
	Real &x2max = mbsize.d_view(m).x2max;
	Real &x3max = mbsize.d_view(m).x3max;
	Real x1v,x2v,x3v;
	x1v = LeftEdgeX(ip, indcs.nx1, x1min, x1max);
	x2v = LeftEdgeX(jp, indcs.nx2, x2min, x2max);
	x3v = LeftEdgeX(kp, indcs.nx3, x3min, x3max);
	Real Dx,Dy,Dz;
	Dx = (x1max - x1min)/indcs.nx1;
	Dy = (x2max - x2min)/indcs.nx2;
	Dz = (x3max - x3min)/indcs.nx3;
	// Interpolate Electric Field at new particle location x1, x2, x3
	E[0] = e0_.x1e(m, kp, jp, ip) + (prtcl_x[0] - x1v)*(e0_.x1e(m, kp, jp, ip+1) - e0_.x1e(m, kp, jp, ip))/Dx;
	E[0] += e0_.x1e(m, kp, jp, ip) + (prtcl_x[1] - x2v)*(e0_.x1e(m, kp, jp+1, ip) - e0_.x1e(m, kp, jp, ip))/Dy;
	E[0] += e0_.x1e(m, kp, jp, ip) + (prtcl_x[2] - x3v)*(e0_.x1e(m, kp+1, jp, ip) - e0_.x1e(m, kp, jp, ip))/Dz;
	E[0] /= 3.0;
	E[1] = e0_.x2e(m, kp, jp, ip) + (prtcl_x[0] - x1v)*(e0_.x2e(m, kp, jp, ip+1) - e0_.x2e(m, kp, jp, ip))/Dx;
	E[1] += e0_.x2e(m, kp, jp, ip) + (prtcl_x[1] - x2v)*(e0_.x2e(m, kp, jp+1, ip) - e0_.x2e(m, kp, jp, ip))/Dy;
	E[1] += e0_.x2e(m, kp, jp, ip) + (prtcl_x[2] - x3v)*(e0_.x2e(m, kp+1, jp, ip) - e0_.x2e(m, kp, jp, ip))/Dz;
	E[1] /= 3.0;
	E[2] = e0_.x3e(m, kp, jp, ip) + (prtcl_x[0] - x1v)*(e0_.x3e(m, kp, jp, ip+1) - e0_.x3e(m, kp, jp, ip))/Dx;
	E[2] += e0_.x3e(m, kp, jp, ip) + (prtcl_x[1] - x2v)*(e0_.x3e(m, kp, jp+1, ip) - e0_.x3e(m, kp, jp, ip))/Dy;
	E[2] += e0_.x3e(m, kp, jp, ip) + (prtcl_x[2] - x3v)*(e0_.x3e(m, kp+1, jp, ip) - e0_.x3e(m, kp, jp, ip))/Dz;
	E[2] /= 3.0;

	// Interpolate Magnetic Field at new particle location x1, x2, x3
	B[0] = b0_.x1f(m, kp, jp, ip) + (prtcl_x[0] - x1v)*(b0_.x1f(m, kp, jp, ip+1) - b0_.x1f(m, kp, jp, ip))/Dx;
	B[0] += b0_.x1f(m, kp, jp, ip) + (prtcl_x[1] - x2v)*(b0_.x1f(m, kp, jp+1, ip) - b0_.x1f(m, kp, jp, ip))/Dy;
	B[0] += b0_.x1f(m, kp, jp, ip) + (prtcl_x[2] - x3v)*(b0_.x1f(m, kp+1, jp, ip) - b0_.x1f(m, kp, jp, ip))/Dz;
	B[0] /= 3.0;
	B[1] = b0_.x2f(m, kp, jp, ip) + (prtcl_x[0] - x1v)*(b0_.x2f(m, kp, jp, ip+1) - b0_.x2f(m, kp, jp, ip))/Dx;
	B[1] += b0_.x2f(m, kp, jp, ip) + (prtcl_x[1] - x2v)*(b0_.x2f(m, kp, jp+1, ip) - b0_.x2f(m, kp, jp, ip))/Dy;
	B[1] += b0_.x2f(m, kp, jp, ip) + (prtcl_x[2] - x3v)*(b0_.x2f(m, kp+1, jp, ip) - b0_.x2f(m, kp, jp, ip))/Dz;
	B[1] /= 3.0;
	B[2] = b0_.x3f(m, kp, jp, ip) + (prtcl_x[0] - x1v)*(b0_.x3f(m, kp, jp, ip+1) - b0_.x3f(m, kp, jp, ip))/Dx;
	B[2] += b0_.x3f(m, kp, jp, ip) + (prtcl_x[1] - x2v)*(b0_.x3f(m, kp, jp+1, ip) - b0_.x3f(m, kp, jp, ip))/Dy;
	B[2] += b0_.x3f(m, kp, jp, ip) + (prtcl_x[2] - x3v)*(b0_.x3f(m, kp+1, jp, ip) - b0_.x3f(m, kp, jp, ip))/Dz;
	B[2] /= 3.0;

}

//----------------------------------------------------------------------------------------
//! \fn  void Lorentz_Terms
//  \brief
KOKKOS_INLINE_FUNCTION
void GRLorentz_Terms( const Real * x_s, const Real * u_s, const Real * E, const Real * B, 
     const bool is_minkowski, const Real spin, const Real q_over_m,
     Real * RHS ){
	
  // Co-variant 4-velocity in the coordinate frame
  // To have compatibility with GR the velocity stored should be the covariant one
  Real u_cov[3];
  Real x[3]; // Position.
  for (int i = 0; i<3; ++i) {
    u_cov[i] = u_s[i];
    x[i] = x_s[i];
  }
  // Get metric components at starting location x1,x2,x3
  Real glower[4][4], gupper[4][4], adm[3][3]; // Metric 
                 // (remember: sqrt(-1/gupper[0][0]) = alpha)
  ComputeMetricAndInverse(x[0],x[1],x[2], is_minkowski, spin, glower, gupper); 
  // Compute 3x3 ADM spatial metric from metric 
  GetUpperAdmMetric( gupper, adm );
  // Determinant of metric needed for vector products
  Real adm_det; 
  ComputeDeterminant3( adm, adm_det );
  // Determinant needed is that of covariant metric
  adm_det = 1.0/adm_det;
  adm_det = sqrt(adm_det);

  // Electric field is stored in fluid frame, need to combine with B to operate on velocity
  Real a_vec[3] = {
    - gupper[0][2]/gupper[0][0]*B[2] + B[1]*gupper[0][3]/gupper[0][0],
    - gupper[0][3]/gupper[0][0]*B[0] + B[2]*gupper[0][1]/gupper[0][0],
    - gupper[0][1]/gupper[0][0]*B[1] + B[0]*gupper[0][2]/gupper[0][0]
  };
  for (int i = 0; i < 3; ++i ){ a_vec[i] *= adm_det; }
  Real u_con[3] = {0.0};
  for (int i1 = 0; i1 < 3; ++i1 ){ 
    for (int i2 = 0; i2 < 3; ++i2 ){ 
    u_con[i1] += adm[i1][i2]*a_vec[i2];
    }
  }
  //Lower indeces of E to covariant for velocity push
  a_vec[0] = E[0] - u_con[0];
  a_vec[1] = E[1] - u_con[1];
  a_vec[2] = E[2] - u_con[2];
  Real push[3];
  push[0] = glower[1][1]*a_vec[0] + glower[1][2]*a_vec[1] + glower[1][3]*a_vec[2];
  push[1] = glower[2][1]*a_vec[0] + glower[2][2]*a_vec[1] + glower[2][3]*a_vec[2];
  push[2] = glower[3][1]*a_vec[0] + glower[3][2]*a_vec[1] + glower[3][3]*a_vec[2];

  Real u0;
  //Lorentz factor in Normal frame
  u0 = adm[0][0]*SQR(u_cov[0]) + adm[1][1]*SQR(u_cov[1]) + adm[2][2]*SQR(u_cov[2])
    + 2.0*adm[0][1]*u_cov[0]*u_cov[1] + 2.0*adm[0][2]*u_cov[0]*u_cov[2] + 2.0*adm[1][2]*u_cov[1]*u_cov[2];
  // In principle the 1.0 should be replaced by a 0 if
  // the particle is massless, but I don't know of 
  // any massless particle that can interact with an 
  // electromagnetic field (unless one goes into quantum mechanics)
  // Convert to Lorentz factor in coordinate frame i.e u^0
  u0 = sqrt(1.0 + u0)*sqrt(-gupper[0][0]);

  //Lower indeces to covariant
  for (int i1 = 0; i1 < 3; ++i1 ){ 
    u_con[i1] = 0.0;
    for (int i2 = 0; i2 < 3; ++i2 ){ 
    u_con[i1] += adm[i1][i2]*u_cov[i2];
    }
    u_con[i1] /= u0;
  }

  Real push2[3] = {
    u_con[1]*B[2] - u_con[2]*B[1],
    u_con[2]*B[0] - u_con[0]*B[2],
    u_con[0]*B[1] - u_con[1]*B[0]
  };
  // Used a vector product, correct for volume
  for (int i = 0; i < 3; ++i ){ push2[i] *= adm_det; }

  RHS[0] += q_over_m*(push[0] + push2[0]);
  RHS[1] += q_over_m*(push[1] + push2[1]);
  RHS[2] += q_over_m*(push[2] + push2[2]);

	return;
}

//----------------------------------------------------------------------------------------
//! \fn  void PushPosition
//  \brief Basic position push RHS
KOKKOS_INLINE_FUNCTION
void GRRHSPosition( const Real * x, const Real * u, const bool is_minkowski, const Real spin, Real * RHS ){
	
  Real glower[4][4], gupper[4][4], adm[3][3]; // Metric 
  Real gamma;
  ComputeMetricAndInverse(x[0],x[1],x[2], is_minkowski, spin, glower, gupper); 
  GetUpperAdmMetric( gupper, adm );
  gamma = adm[0][0]*SQR(u[0]) + adm[1][1]*SQR(u[1]) + adm[2][2]*SQR(u[2])
    + 2.0*adm[0][1]*u[0]*u[1] + 2.0*adm[0][2]*u[0]*u[2] + 2.0*adm[1][2]*u[1]*u[2];
  gamma = sqrt(1.0 + gamma)*sqrt(-gupper[0][0]);
  RHS[0] = adm[0][0]*u[0] + adm[0][1]*u[1] + adm[0][2]*u[2];
  RHS[1] = adm[1][0]*u[0] + adm[1][1]*u[1] + adm[1][2]*u[2];
  RHS[2] = adm[2][0]*u[0] + adm[2][1]*u[1] + adm[2][2]*u[2];
  RHS[0] /= gamma;
  RHS[1] /= gamma;
  RHS[2] /= gamma;
  RHS[0] += gupper[0][1]/gupper[0][0];
  RHS[1] += gupper[0][2]/gupper[0][0];
  RHS[2] += gupper[0][3]/gupper[0][0];

	return;
}

//----------------------------------------------------------------------------------------
//! \fn  void PushPosition
//  \brief Basic position push RHS
KOKKOS_INLINE_FUNCTION
void GRRHSVelocity( const Real * x, const Real * u, const bool is_minkowski, const Real spin, Real * RHS ){
	
  Real gl[4][4], gu[4][4], adm[3][3]; // Metric 
  Real dg_dx1[4][4], dg_dx2[4][4], dg_dx3[4][4]; //Metric derivatives
  Real gamma;

  ComputeMetricAndInverse(x[0],x[1],x[2], is_minkowski, spin, gl, gu); 
  GetUpperAdmMetric( gu, adm );
  gamma = adm[0][0]*SQR(u[0]) + adm[1][1]*SQR(u[1]) + adm[2][2]*SQR(u[2])
    + 2.0*adm[0][1]*u[0]*u[1] + 2.0*adm[0][2]*u[0]*u[2] + 2.0*adm[1][2]*u[1]*u[2];
  gamma = sqrt(1.0 + gamma);
  bool get_upper = true;
  ComputeMetricDerivatives(x[0], x[1], x[2], is_minkowski, spin, get_upper,
                              dg_dx1, dg_dx2, dg_dx3);
  Real (*dg_array[3])[4][4] = {&dg_dx1, &dg_dx2, &dg_dx3};

  for (int dir = 0; dir<3; ++dir){
    Real (&use_dx)[4][4] = *dg_array[dir]; 

    Real dx_alpha = 0.5*( use_dx[0][0] ); 
    dx_alpha /= ( sqrt(pow(-gu[0][0],3)) ); 

    Real dx_beta[3];
    for (int ii=0; ii<3; ++ii){
      dx_beta[ii] = -( use_dx[0][ii+1] )/( gu[0][0] );
      dx_beta[ii] += gu[0][ii+1]*( use_dx[0][0] )/( SQR(gu[0][0]) );
    }

    Real dx_gamma[3][3];
    for (int ii=0; ii<3; ++ii){
      for (int ij=0; ij<3; ++ij){
        dx_gamma[ii][ij] = ( use_dx[ii+1][ij+1] );
        dx_gamma[ii][ij] -= 2.0*gu[0][ij+1]*( use_dx[0][ii+1] )/gu[0][0];
        dx_gamma[ii][ij] += gu[0][ij+1]*gu[0][ii+1]*( use_dx[0][0] )/SQR(gu[0][0]);
      }
    }

    RHS[dir] = -dx_alpha*gamma;
    RHS[dir] += dx_beta[0]*u[0] + dx_beta[1]*u[1] + dx_beta[2]*u[2];
    for (int ii=0; ii<3; ++ii){
      for (int ij=0; ij<3; ++ij){
        RHS[dir] -= 0.5*dx_gamma[ii][ij]*u[ii]*u[ij]/( gamma*sqrt(-gu[0][0]) );
      }
    }
  }

	return;
}

//----------------------------------------------------------------------------------------
//! \fn  void LUDecomposition
//  \brief Compute Lower and Upper triangular metrices from original matrix
KOKKOS_INLINE_FUNCTION
void LUDecomposition( const int ndim, Real * LUMat, int * perm ){

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
//  \brief Compute Lower and Upper triangular metrices from original matrix
KOKKOS_INLINE_FUNCTION
void FWDSubstitution(const int ndim, const Real * LUMat, const int * perm, const Real * idArr, Real * outArr) {

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
//  \brief Compute Lower and Upper triangular metrices from original matrix
KOKKOS_INLINE_FUNCTION
void BWDSubstitution(const int ndim, const Real * LUMat, const int * perm, const Real * inArr, Real * outArr) {

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
//  \brief Compute the inverse of an nxn matrix
KOKKOS_INLINE_FUNCTION
void InvertMatrixLU( const int ndim, const Real * inputMat, Real * outputMat ){

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

  LUDecomposition(ndim, LUMat, perm_arr);

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
