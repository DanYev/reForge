"""
Asynchronous OpenMM Reporter for heavy calculations that don't throttle MD engine.
"""
import threading
import queue
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class AsyncHeavyReporter:
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
        data = {
            'step': simulation.currentStep,
            'time': state.getTime().value_in_unit_system(simulation.system.getDefaultLengthUnit()),
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
