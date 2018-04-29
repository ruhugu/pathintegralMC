#pragma once
#include "dranxor2/dranxor2C.h"
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>

// Update the trajectory the given number of Monte Carlo steps.
int c_update_traj(
        double* traj_ini, int ncoords, int nsteps,
        double jump_radius, int nprops, double timestep, 
        double (*potential)(double, double*), double* potential_args,
        double* traj_out);

// Calculate the action terms involving a certain timestep position.
double c_localAction(
        double xprev, double x, double xnext, double timestep,
        double (*potential)(double, double*), double* potential_args);

// Return the potential energy of the harmonic oscillator.
double c_harmonic_potential(double x, double* args);

// Given 2D indices, return the corresponding index in the flattened array
int index2D(int row, int column, int rowlength);
