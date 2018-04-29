import numpy as np
import random
import cmodule


class Trajectories(object):
    """Class for generation and manipulation of quantum trajectories.

    """
    def __init__(self, x_ini, x_fin, t_ini, t_fin, timestep=0.1, seed=None):
        """Initialization method.
        
        Parameters
        ----------
            x_ini : float
                Initial position of the trajectory.

            x_fin : float
                Final position of the trajectory.

            t_ ini : float
                Initial time of the trajectory. 

            t_fin : float
                Final time of the trajectory.

            timestep : float
                Time step.

            traj_ini : float array
                Initial trajectory. If None, an initial trajectory is
                assumed.

        """
        # Store parameters
        self.x_ini = x_ini
        self.x_fin = x_fin
        self.t_ini = t_ini
        self.t_fin = t_fin

        # If the time interval is not a multiple of timestep, 
        # change the timestep.
        nsteps = float(t_fin - t_ini)/timestep
        if not nsteps.is_integer():
            nsteps = int(nsteps) + 1
            timestep = float(t_fin - t_ini)/nsteps

        self.nsteps = int(nsteps)
        self.timestep = timestep

        # Calculate initial trajectory (assuming it is linear).
        # trajs is a 2D array.
        self.trajs = np.array((np.linspace(x_ini, x_fin, nsteps, dtype=float),))
        
        # Initialize the random number generators
        if seed == None:
            seed = random.SystemRandom().randint(0, 32767)
        self.seed = seed  # Store seed
        cmodule.seed(seed)  # dranxor number generator
        np.random.seed(seed)  # Python random number generator


    def add_trajectories(
            self, nmeasures, measure_interval=50, nprops=10, jump_radius=1./20,
            relaxtime=1000, traj_ini=None):
        """Calculate and store new trajectories.
            
        Parameters
        ----------
            nmeasures : int
                Number of new trajectories to be stored.

            measure_interval : int
                Number of MC steps between measures.

            nprops : int
                Number of consecutive Metropolis steps for each coordinate.

            jump_radius : float
                Maximum difference in the coordinate for consecutive 
                Metropolis steps.

            relaxtime : int
                Number of MC steps before storing trajectories.

            traj_ini : float array
                 Initial trajectory. If None, the last stored trajectory is used.

        """
        # Initialize the trajectory
        if traj_ini == None:
            traj = np.copy(self.trajs[-1])
        else:
            traj = np.copy(traj_ini)

        # Create array to store the generated trajectories
        newtrajs = np.zeros((nmeasures, self.nsteps), dtype=float)

        # Let the trajectory relax
        self.update_traj(traj, relaxtime, nprops, jump_radius)

        # Measure
        for j_measure in range(nmeasures):
            traj, n_accepted = self.update_traj(
                    traj, relaxtime, nprops, jump_radius)
            newtrajs[j_measure] = traj

        # Store new measures
        self.trajs = np.vstack((self.trajs, newtrajs))

        return


    def update_traj(self, traj_ini, nsteps, nprops, jump_radius):
        """Update trajectory.

        This is a placeholder.

        Parameters
        ----------
            traj_ini : float array
                Initial state of the trajectory.

            nsteps : int
                Number of MC steps.

            nprops : int
                Number of consecutive Metropolis steps for each coordinate.

            jump_radius : float
                Maximum difference in the coordinate for consecutive 
                Metropolis steps.

            relaxtime : int
                Number of MC steps before storing trajectories.

            traj_ini : float array
                 Initial trajectory. If None, the last stored trajectory is used.

        Returns 
        -------
            traj_out : float array
                Updated trajectory.

            n_accepted : int
                Number of accepted proposals.

        """
        traj_out = np.copy(traj_ini)
        n_accepted = 0

        return traj_out, n_accepted


class HarmonicOscillator(Trajectories):
    """

    """
    def update_traj(self, traj_ini, nsteps, nprops, jump_radius):
        """Update trajectory.

        This is a placeholder.

        Parameters
        ----------
            traj_ini : float array
                Initial state of the trajectory.

            nsteps : int
                Number of MC steps.

            nprops : int
                Number of consecutive Metropolis steps for each
                coordinate.

            jump_radius : float
                Maximum difference in the coordinate for consecutive 
                Metropolis steps.

            relaxtime : int
                Number of MC steps before storing trajectories.

            traj_ini : float array
                 Initial trajectory. If None, the last stored
                 trajectory is used.

        Returns 
        -------
            traj_out : float array
                Updated trajectory.

            n_accepted : int
                Number of accepted proposals.

        """
        traj_out, n_accepted = cmodule.update_traj_harmonic(
                traj_ini, nsteps, jump_radius, nprops, self.timestep)

        return traj_out, n_accepted
