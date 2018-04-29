#include "c_module.h"
#include <stdio.h>


/* Update the trajectory the given number of Monte Carlo steps.

 Parameters
 ----------
     traj_ini  
         Initial traectory. 1D array with the position in each
         timestep.
     ncoords
         Number of elements in traj_ini.

     nsteps
         Number of Monte Carlo steps.

     jump_radius 
         Max difference in the coordinate for consecutive 
         Metropolis steps.

     nprops
         Number of consecutive Metropolis steps for each coordinate.

     timestep
        Timestep of the simulation.

     potential
         Potential function.

     potential_args 
         Potential function arguments array.

 Output parameters
 -----------------
     traj_out
         The final trajectory will be stored here.

*/
int c_update_traj(
        double* traj_ini, int ncoords, int nsteps,
        double jump_radius, int nprops, double timestep, 
        double (*potential)(double, double*), double* potential_args,
        double* traj_out)
{
    // Variable declaration
    int n_accepted; // Number of accepted numbers
    double local_action, last_local_action; 
    double newcoord;
    double accept_prob;
    bool prop_accepted;
    int j_step, j_coord, j_prop; // Counters

    // Initialize the number of accepted numbers
    n_accepted = 0;

    // Initialize traj to the initial trajetory traj_ini
    for (j_coord = 0; j_coord < ncoords; j_coord++)
    {
        traj_out[j_coord] = traj_ini[j_coord];
    }

    for (j_step = 0; j_step < nsteps; j_step++)
    {
        for (j_coord = 1; j_coord < ncoords - 1; j_coord++)
        {
            // Initialize last_local_action
            last_local_action = c_localAction(
                    traj_out[j_coord-1], traj_out[j_coord], traj_out[j_coord+1],
                    timestep, potential, potential_args);

            for (j_prop = 0; j_prop < nprops; j_prop++)
            {
                // Generate proposal to the considered coordinate
                newcoord = traj_out[j_coord] + jump_radius*(dranu_() - 1);

                // Calculate the local action for the new value
                local_action = c_localAction(
                        traj_out[j_coord-1], newcoord, traj_out[j_coord+1], 
                        timestep, potential, potential_args);

                // Check if the proposal is accepted directly
                // (a smaller action means a bigger probability density)
                prop_accepted = (local_action < last_local_action);

                // If not, accept according to the acceptance probability
                if (!prop_accepted)
                {
                    // The acceptance probability is the quotient of the 
                    // probability densities, which equals the exponential
                    // of the opposite of the action change.
                    accept_prob = exp(-(local_action - last_local_action));
                    prop_accepted = (dranu_() < accept_prob);
                }

                // If accepted, update coordinate
                if (prop_accepted)
                {
                    traj_out[j_coord] = newcoord;
                    n_accepted++;
                    last_local_action = local_action;
                }
            }
        }
    }

    return n_accepted;
}


/* Calculate the action terms involving a certain timestep position.
 
  The units are chosen so that the particle mass is 1.

  Parameters
  ----------

     xprev 
         Position in the previous timestep.

     x
         Position in the timestep.

     xnext
         Position in the next timestep.

     timestep
         Time difference between consecutive timesteps.

     potential
         Potential function. It must depend only on the position.

     potential_args 
         Potential function arguments array.

*/
double c_localAction(
        double xprev, double x, double xnext, double timestep,
        double (*potential)(double, double*), double* potential_args)
{
    return timestep*(
            (0.5*pow((x - xprev)/timestep, 2) + potential(xprev, potential_args))
            + (0.5*pow((xnext - x)/timestep, 2) + potential(x, potential_args)));
}


/* Return the potential energy of the harmonic oscillator.
 
   The units are taken so that both the mass of the particle and the
   frequency of the oscillator equal 1.
 
*/
double c_harmonic_potential(double x, double* args)
{
    return 0.5*x*x;
}


// Given 2D indices, return the corresponding index in the flattened array
int index2D(int row, int column, int rowlength)
{
    return row*rowlength + column;
}            

