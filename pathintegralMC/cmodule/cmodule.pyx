# -*- coding: utf-8 -*-

# Python extension through Cython following:
# http://www.scipy-lectures.org/advanced/interfacing_with_c/interfacing_with_c.html#id10
# Nice example:
# https://gist.github.com/phaustin/4973792

from __future__ import (print_function, division, 
                        absolute_import, unicode_literals)

# Imports Cython declarations for numpy.
# cimport is used to import C data types, functions, etc defined in 
# other Cython file. Details: 
# http://cython.readthedocs.io/en/latest/src/userguide/sharing_declarations.html#the-cimport-statement
# In this case we are not importing Python numpy. We are loading the 
# Cython code that allows it to interact with numpy.
cimport numpy as cnp

# Enable Numpy-C-API access.
# Interesting:
# https://github.com/cython/cython/wiki/tutorials-numpy
# Numpy-C-API:
# https://docs.scipy.org/doc/numpy/reference/c-api.html
#np.import_array()
import numpy as np

# This tells Cython that there the following functions are defined 
# elsewhere and their header is in "cmodule.h".
# cdef is used to define c functions.
# http://notes-on-cython.readthedocs.io/en/latest/function_declarations.html
# cdef extern especifies that the function is defined elsewhere.
# http://cython-docs2.readthedocs.io/en/latest/src/userguide/external_C_code.html
cdef extern from "c_module.h":
    int c_update_traj(
            double* traj_ini, int ncoords, int nsteps,
            double jump_radius, int nprops, double timestep,
            double (*potential)(double, double*), double* potential_args,
            double* traj)
    double c_harmonic_potential(double x, double* args)

# More of the same for the random generator
cdef extern from "dranxor2/dranxor2C.h":
    void dranini_(int*)


# Define a new c type for a function taking the arguments of potential
# function. This is needed for passing functions as arguments in Cython
# functions.
ctypedef double (*potential_fun)(double, double*)

# Define a wrapper function that will act as a bridge between Python and 
# the C function <--- no se hasta que punto es esto totalmente cierto
# not None: by default, Cython allows arguments that meet the specified
# data type or that are None. In order to prevent the last behaviour, we 
# must add not None after the parameter name.
# http://docs.cython.org/en/latest/src/userguide/extension_types.html#extension-types-and-none

# I can't find where the sintax np.ndarray[...] is explained. However, from 
# this example we can see that the first argument is the datatype, ndim 
# refers to the dimension of the array and I think mode determines how 
# the array is stored in the memory. In this case we would use the c-way
# (whatever that is). What I have found about the matter:
# Here they say it is realated to "efficient buffer access". Other modes are
# also presented.
# https://github.com/cython/cython/wiki/enhancements-buffer
# The <...> before argument are type casts:
# http://cython.readthedocs.io/en/latest/src/reference/language_basics.html#type-casting
# If we use a Python variable as an argument of a Cython function with a
# specified type, automatic conversion will be attempted.
# http://cython.readthedocs.io/en/latest/src/userguide/language_basics.html

# ALTERNATIVE: MemoryViews
# http://nealhughes.net/cython1/
# http://docs.cython.org/en/latest/src/userguide/memoryviews.html


cdef update_traj(
        cnp.ndarray[double, ndim=1, mode="c"] traj_ini,
        int nsteps, double jump_radius, int nprops, 
        timestep, potential_fun potential, 
        cnp.ndarray[double, ndim=1, mode="c"] potential_args):
    """Update trajetory in the given number of MC steps.
        
    Parameters
    ----------
        traj_ini : float array
             Initial traectory. 1D array with the position in each
             timestep.

        nsteps : int
            Number of MC steps.

        jump_radius : float
            Max difference in the coordinate for consecutive 
            Metropolis steps.

        nprops : int
            Number of consecutive Metropolis steps for each coordinate.

        timestep : double
            Timestep used in the simulation.

        potential : function 
            Potential function. 

        potential_args : float array
            Potential function arguments array.

    Returns
    -------
        traj_out : float array
            Updated trajectory.

        naccept : Number of accepted proposals.

    """
    ncoords = traj_ini.size

    # Create vector to store the final trajectory
    cdef cnp.ndarray[double, ndim=1, mode="c"] traj_out = np.zeros(
            ncoords, dtype="float64")

    # Call the C function
    naccept = c_update_traj(
            <double*> cnp.PyArray_DATA(traj_ini),
            ncoords, nsteps, jump_radius, nprops,
            timestep, potential,
            <double*> cnp.PyArray_DATA(potential_args),
            <double*> cnp.PyArray_DATA(traj_out))

    return traj_out, naccept


def update_traj_harmonic(
        cnp.ndarray[double, ndim=1, mode="c"] traj_ini,
        int nsteps, double jump_radius, int nprops, double timestep):
    """Update trajetory with an harmonic osicillator potential.
        
    Parameters
    ----------
        traj_ini : float array
             Initial traectory. 1D array with the position in each
             timestep.

        nsteps : int
            Number of MC steps.

        jump_radius : float
            Max difference in the coordinate for consecutive 
            Metropolis steps.

        nprops : int
            Number of consecutive Metropolis steps for each coordinate.

    Returns
    -------
        traj_out : float array
            Updated trajectory.

        naccept : int
            Number of accepted proposals.

    """
    # The harmonic potential function does not need additional
    # arguments. Therefore create null pointer as placeholder.
    cdef cnp.ndarray[double, ndim=1, mode="c"] args = np.zeros(
            1, dtype="float64")

    # Call the generic update function using the harmonic potential.
    traj_out, naccept = update_traj(
            traj_ini, nsteps, jump_radius, nprops, timestep,
            c_harmonic_potential, args)

    return traj_out, naccept


def seed(int iseed): 
    """Initialize the random number generator in the c extension.
    
    Parameters
    ----------
        iseed : int
            Seed for the random number generator.

    """
    dranini_(&iseed)
    return    
