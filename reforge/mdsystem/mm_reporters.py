"""
Asynchronous OpenMM Reporter for heavy calculations that don't throttle MD engine.
"""
import threading
import queue
import numpy as np
from pathlib import Path
import logging
import MDAnalysis as mda
from openmm import unit
from MDAnalysis.lib.mdamath import triclinic_box

logger = logging.getLogger(__name__)

class CustomReporter(object):
    def __init__(
        self,
        file,
        reportInterval,
        enforcePeriodicBox=None,
        selection: str = None,
        writer_kwargs: dict = None
    ):
        self._reportInterval = reportInterval
        self._enforcePeriodicBox = enforcePeriodicBox
        self._filename = file
        self._topology = None
        self._nextModel = 0
        self._mdaUniverse = None
        self._mdaWriter = None
        self._selection = selection
        self._atomGroup = None
        self._writer_kwargs = writer_kwargs or {}

    def describeNextReport(self, simulation):
        """Get information about the next report this object will generate.
        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for
        Returns
        -------
        tuple
            A six element tuple. The first element is the number of steps
            until the next report. The next four elements specify whether
            that report will require positions, velocities, forces, and
            energies respectively.  The final element specifies whether
            positions should be wrapped to lie in a single periodic box.
        """
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        positions, velocities, forces = True, True, True
        return steps, positions, velocities, forces, False, self._enforcePeriodicBox

    def makeUniverse(self, simulation):
        dt = simulation.integrator.getStepSize() * self._reportInterval # Time between frames in ps
        self._mdaUniverse = mda.Universe(
            simulation.topology,
            simulation,
            topology_format='OPENMMTOPOLOGY',
            format='OPENMMSIMULATION',
            in_memory_step=self._reportInterval,
            in_memory=True,
        )
        if self._selection is not None:
            self._atomGroup = self._mdaUniverse.select_atoms(self._selection)
        else:
            self._atomGroup = self._mdaUniverse.atoms
        
    def getState(self, state):
        self.positions = state.getPositions(asNumpy=True)
        self.velocities = state.getVelocities(asNumpy=True)
        self.boxVectors = state.getPeriodicBoxVectors(asNumpy=True)
        self.sim_time = state.getTime().value_in_unit(unit.picosecond)

    def updateUniverse(self):
        positions = self.positions.value_in_unit(unit.angstrom)
        self._mdaUniverse.atoms.positions = positions
        velocities = self.velocities.value_in_unit(unit.angstrom/unit.picosecond)
        self._mdaUniverse.atoms.velocities = velocities
        boxVectors = self.boxVectors.value_in_unit(unit.angstrom)
        self._mdaUniverse.dimensions = triclinic_box(*boxVectors)
        self._mdaUniverse.dimensions[:3] = self._sanitize_box_angles(self._mdaUniverse.dimensions[:3])
        self._mdaUniverse.trajectory.ts.time = self.sim_time
        self._mdaUniverse.trajectory.ts.frame = self._nextModel - 1

    def report(self, simulation, state):
        """Generate a report.
        
        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for
        state : State
            The current state of the simulation
        """
        if self._nextModel == 0:
            self.makeUniverse(simulation)
            self._nextModel += 1
        self.getState(state)
        self.updateUniverse()
        ag = self._atomGroup
        data = ag.ts, ag.positions
        print(data)
        self._nextModel += 1

    @staticmethod
    def _sanitize_box_angles(angles):
        """ Ensure box angles correspond to first quadrant
        See `discussion on unitcell angles <https://github.com/MDAnalysis/mdanalysis/pull/2917/files#r620558575>`_
        """
        inverted = 180 - angles
        return np.min(np.array([angles, inverted]), axis=0)


############################################################################################# 
### Trajectory conversion and fitting ###
#############################################################################################

def convert_trajectories(topology, trajectories, out_topology, out_trajectory, ref_top=None,
        selection="name CA", step=1, skip_selection=False, fit=True):
    """
    Convert and fit trajectories using MDAnalysis.
    
    Parameters:
    - topology: str, path to the input topology file (e.g., PDB)
    - trajectories: list of str, paths to input trajectory files (e.g., XTC, TRR)
    - out_topology: str, path to the output topology file
    - out_trajectory: str, path to the output trajectory file
    - selection: str, atom selection string for MDAnalysis
    - step: int, step interval for saving frames
    - fit: bool, whether to fit the trajectory to the reference structure
    """
    tmp_traj = Path(out_trajectory).parent / ('temp_traj' + out_trajectory.suffix)
    logger.info(f'Converting trajectory with selection: {selection}')
    _trjconv_selection(trajectories, topology, tmp_traj, out_topology, 
        selection=selection, step=step)
    if fit:
        logger.info('Fitting trajectory to reference structure')
        transform_vels = str(out_trajectory).endswith('.trr') # True for .trr files
        _trjconv_fit(tmp_traj, out_topology, out_trajectory, 
            ref_top=ref_top, selection=selection, transform_vels=transform_vels)
        os.remove(tmp_traj)
    else:
        os.rename(tmp_traj, out_trajectory)


def _trjconv_selection(input_traj, input_top, output_traj, output_top, selection="name CA", step=1):
    u = mda.Universe(input_top, input_traj)
    selected_atoms = u.select_atoms(selection)
    n_atoms = selected_atoms.n_atoms
    selected_atoms.write(output_top)
    with mda.Writer(str(output_traj), n_atoms=n_atoms) as writer:
        for ts in u.trajectory[::step]:
            writer.write(selected_atoms)
            if ts.frame % 1000 == 0:
                frame = ts.frame
                time_ns = ts.time / 1000
                logger.info(f"Current frame: %s at %s ns", frame, time_ns)
    logger.info("Saved selection '%s' to %s and topology to %s", selection, output_traj, output_top)


def _trjconv_fit(input_traj, input_top, output_traj, ref_top=None, selection='name CA', transform_vels=False):
    u = mda.Universe(input_top, input_traj)
    ag = u.select_atoms(selection)
    if not ref_top:
        ref_top = input_top
    ref_u = mda.Universe(ref_top) 
    ref_ag = ref_u.select_atoms(selection)
    u.trajectory.add_transformations(fit_rot_trans(ag, ref_ag,))
    logger.info("Converting/Writing Trajecory")
    with mda.Writer(str(output_traj), ag.n_atoms) as W:
        for ts in u.trajectory:   
            if transform_vels:
                transformed_vels = _tranform_velocities(ts.velocities, ts.positions, ref_ag.positions)
                ag.velocities = transformed_vels
            W.write(ag)
            if ts.frame % 1000 == 0:
                frame = ts.frame
                time_ns = ts.time / 1000
                logger.info(f"Current frame: %s at %s ns", frame, time_ns)
    logger.info("Done!")


def _tranform_velocities(vels, poss, ref_poss):
    R = _kabsch_rotation(poss, ref_poss)
    vels_aligned = vels @ R
    return vels_aligned
    

def _kabsch_rotation(P, Q):
    """
    Return the 3x3 rotation matrix R that best aligns P onto Q (both Nx3),
    after removing centroids (i.e., pure rotation via Kabsch).
    """
    # subtract centroids
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    # covariance and SVD
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # right-handed correction
    if np.linalg.det(R) < 0.0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    return R 

############################################################################################# 
### Utility functions ###
#############################################################################################

def get_platform_info():
    """Report OpenMM platform and hardware information."""
    info = {}
    # Get number of available platforms and their names
    num_platforms = mm.Platform.getNumPlatforms()
    info['available_platforms'] = [mm.Platform.getPlatform(i).getName() 
                                 for i in range(num_platforms)]
    # Try to get the fastest platform (usually CUDA or OpenCL)
    platform = None
    for platform_name in ['CUDA', 'OpenCL', 'CPU']:
        try:
            platform = mm.Platform.getPlatformByName(platform_name)
            info['platform'] = platform_name
            break
        except Exception:
            continue 
    if platform is None:
        platform = mm.Platform.getPlatform(0)
        info['platform'] = platform.getName()
    # Get platform properties
    info['properties'] = {}
    try:
        if info['platform'] in ['CUDA', 'OpenCL']:
            info['properties']['device_index'] = platform.getPropertyDefaultValue('DeviceIndex')
            info['properties']['precision'] = platform.getPropertyDefaultValue('Precision')
            if info['platform'] == 'CUDA':
                info['properties']['cuda_version'] = mm.version.cuda
            info['properties']['gpu_name'] = platform.getPropertyValue(platform.createContext(), 'DeviceName')
        info['properties']['cpu_threads'] = platform.getPropertyDefaultValue('Threads')
    except Exception as e:
        logger.warning(f"Could not get some platform properties: {str(e)}")
    # Get OpenMM version
    info['openmm_version'] = mm.version.full_version
    # Log the information
    logger.info("OpenMM Platform Information:")
    logger.info(f"Available Platforms: {', '.join(info['available_platforms'])}")
    logger.info(f"Selected Platform: {info['platform']}")
    logger.info(f"OpenMM Version: {info['openmm_version']}")
    logger.info("Platform Properties:")
    for key, value in info['properties'].items():
        logger.info(f"  {key}: {value}")
    return info


class AsyncReporter:
    """
    OpenMM reporter that performs heavy calculations in a background thread.
    
    The report() method quickly copies necessary data and returns immediately,
    allowing the MD engine to continue. Heavy calculations happen in parallel
    in a background worker thread.
    
    Example:
        reporter = AsyncHeavyReporter('output.dat', reportInterval=1000,
                                     calculation_func=my_heavy_function)
        simulation.reporters.append(reporter)
        simulation.step(100000)
        reporter.finalize()  # Wait for all calculations to complete
    """
    
    def __init__(self, output_file, reportInterval, calculation_func=None, 
                 queue_size=20, **kwargs):
        """
        Parameters
        ----------
        output_file : str or Path
            Output file path for results
        reportInterval : int
            Report every N steps
        calculation_func : callable
            Function that takes data dict and returns result
            Signature: func(data: dict) -> result
        queue_size : int
            Maximum number of frames to buffer (default: 20)
        **kwargs : dict
            Additional arguments passed to calculation_func
        """
        self._reportInterval = reportInterval
        self._output_file = Path(output_file)
        self._calculation_func = calculation_func or self._default_calculation
        self._calc_kwargs = kwargs
        
        # Thread-safe queue for data
        self._queue = queue.Queue(maxsize=queue_size)
        
        # Background worker thread
        self._running = True
        self._worker = threading.Thread(target=self._worker_loop, daemon=False)
        self._worker.start()
        
        # Results storage
        self._results = []
        self._lock = threading.Lock()
        
        logger.info(f"AsyncHeavyReporter initialized with queue_size={queue_size}")
    
    def describeNextReport(self, simulation):
        """
        Tell OpenMM what data we need.
        
        Returns
        -------
        tuple : (steps, positions, velocities, forces, energies, wrap)
        """
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        # Request positions, velocities, forces (adjust as needed)
        needPositions = True
        needVelocities = True
        needForces = True
        needEnergies = False
        enforcePeriodicBox = False
        
        return (steps, needPositions, needVelocities, needForces, 
                needEnergies, enforcePeriodicBox)
    
    def report(self, simulation, state):
        """
        Called by OpenMM - MUST BE FAST!
        
        Just copy data and queue it, then return immediately.
        """
        # Quickly copy data (don't reference, as state object will be reused)
        dt = simulation.integrator.getStepSize() * self._reportInterval # Time between frames in ps
        data = {
            'dt': dt,
            'step': simulation.currentStep,
            'time': state.getTime(),
            'positions': state.getPositions(asNumpy=True).copy(),
            'velocities': state.getVelocities(asNumpy=True).copy() if state.getVelocities() is not None else None,
            'forces': state.getForces(asNumpy=True).copy() if state.getForces() is not None else None,
        }
        
        # Try to queue (non-blocking with timeout)
        try:
            self._queue.put(data, timeout=0.5)
            logger.debug(f"Queued frame at step {simulation.currentStep}")
        except queue.Full:
            logger.warning(f"Queue full at step {simulation.currentStep}, frame skipped!")
    
    def _worker_loop(self):
        """
        Background thread that processes queued data.
        This is where heavy calculations happen.
        """
        logger.info("Worker thread started")
        
        while self._running or not self._queue.empty():
            try:
                # Get data with timeout
                data = self._queue.get(timeout=1.0)
                
                # HEAVY CALCULATION HAPPENS HERE - PARALLEL TO MD!
                try:
                    result = self._calculation_func(data, **self._calc_kwargs)
                    
                    # Store result thread-safely
                    with self._lock:
                        self._results.append(result)
                    
                    logger.debug(f"Processed step {data['step']}")
                    
                except Exception as e:
                    logger.error(f"Error in calculation at step {data['step']}: {e}")
                
                finally:
                    self._queue.task_done()
                    
            except queue.Empty:
                continue
        
        logger.info("Worker thread finished")
    
    def _default_calculation(self, data):
        """Default placeholder calculation."""
        # Example: compute RMSD, distance matrix, or whatever
        positions = data['positions']
        # Do something with positions...
        return {'step': data['step'], 'positions': positions}
    
    def finalize(self):
        """
        Wait for all queued calculations to complete and save results.
        Call this after simulation.step() finishes.
        """
        logger.info("Finalizing reporter...")
        
        # Wait for queue to empty
        self._queue.join()
        
        # Stop worker thread
        self._running = False
        self._worker.join(timeout=10.0)
        
        # Save results
        if self._results:
            logger.info(f"Saving {len(self._results)} results to {self._output_file}")
            self._save_results()
        else:
            logger.warning("No results to save")
    
    def _save_results(self):
        """Save accumulated results to file."""
        with self._lock:
            # Save as numpy array or pickle depending on structure
            if isinstance(self._results[0], dict):
                # Save as npz
                result_dict = {k: [r[k] for r in self._results] 
                              for k in self._results[0].keys()}
                np.savez(self._output_file, **result_dict)
            else:
                # Save as npy
                np.save(self._output_file, self._results)
        
        logger.info(f"Results saved to {self._output_file}")
    
    def __del__(self):
        """Cleanup on deletion."""
        if self._running:
            logger.warning("Reporter deleted before finalize() was called!")
            self.finalize()


# Example usage and heavy calculation function
def example_heavy_calculation(data):
    """
    Example of a computationally expensive function.
    This could be:
    - Distance matrix calculation
    - Hydrogen bond analysis  
    - Secondary structure detection
    - Contact map generation
    - etc.
    """
    positions = data['positions']
    com = np.mean(positions, axis=0)
    
    return {
        'step': data['step'],
        'com': com,
        'n_atoms': len(positions)
    }


if __name__ == "__main__":
    # Example usage
    print("AsyncHeavyReporter module loaded")
    print("Use AsyncHeavyReporter in your OpenMM simulations for parallel heavy calculations")
