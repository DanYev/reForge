"""
Example usage of AsyncReporter with OpenMM simulations.
Water box simulation with Mean Square Displacement (MSD) calculation.
"""
import numpy as np
from pathlib import Path
import openmm as mm
import MDAnalysis as mda
from openmm import app, unit

# Import the async reporter
from reforge.mdsystem.mm_reporters import CustomReporter
from graph_utils import live_plot
from ring import vector_ring_buffer

# ============================================================================
# Mean Square Displacement (MSD) Calculation
# ============================================================================

def msd_calculation(data, initial_positions):
    window_size = 250  # Size of ring buffer (250 x 0.008 ps = 2.0 ps)
    buffer = vector_ring_buffer(window_size)  # Ring buffer for 250 x 3D vectors
    ready = False  # Flag to track if ring buffer is full
    # Correlation time axis
    dt = data['dt'].value_in_unit(unit.picoseconds)
    tau_ps = np.arange(0, window_size) * dt
    # Cumulative sum of MSD values for averaging
    msd_sum = np.zeros(window_size)
    n_samples = 0  # Number of MSD samples collected

    u = mda.Universe('topology.pdb')
    sel = u.select_atoms("resid 1")
    # tStep = 0
    # # Main analysis loop
    # for ts in u.trajectory:
    #     com_position = sel.center_of_mass()
    #     buffer/.append(com_position)
    #     if buffer.size == window_size: # Buffer is filled
    #         ready = True
            
    #     # Calculate MSD when buffer is filled
    #     if ready:
    #         history = buffer.get_values() # All positions in ring buffer (sorted)
    #         ref = history[0]  # Reference position 
    #         # Calculate MSD
    #         displacements = history - ref  # Displacements for all times in history
    #         msd_values = np.sum(displacements**2, axis=1) # Convert displacements to MSDs
    #         # Add to cumulative sum
    #         msd_sum += msd_values
    #         n_samples += 1
    #         # Calculate average MSDs 
    #         msd_avg = msd_sum / n_samples
    #     tStep += 1
    #     if tStep % window_size ==0:
    #         plot['update'](tau_ps, msd_avg)

# ============================================================================
# Water Box Simulation with MSD
# ============================================================================

def create_water_box(n_waters=216, box_size=1.86):
    """
    Create a water box system using OpenMM.
    
    Parameters
    ----------
    n_waters : int
        Number of water molecules (default: 216 for ~1.86 nm box)
    box_size : float
        Box size in nanometers
        
    Returns
    -------
    topology : openmm.app.Topology
        System topology
    system : openmm.System
        OpenMM system
    positions : list
        Initial positions with units
    """
    print(f"\nCreating water box with ~{n_waters} molecules...")
    print(f"Box size: {box_size} nm")
    # Create forcefield
    forcefield = app.ForceField('tip3p.xml')
    # Use OpenMM's Modeller to create a water box
    modeller = app.Modeller(app.Topology(), [])
    # Add solvent (water) to fill the box
    # boxSize expects values in nm without units
    modeller.addSolvent(forcefield, model='tip3p', boxSize=[box_size, box_size, box_size]*unit.nanometers)
    topology = modeller.topology
    positions = modeller.positions
    with open('topology.pdb', 'w') as f:
        app.PDBFile.writeFile(topology, positions, f)
    # Create system with TIP3P water model
    system = forcefield.createSystem(
        topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=0.9*unit.nanometers,
        constraints=app.HBonds,
        rigidWater=True,
        ewaldErrorTolerance=0.0005
    )
    n_atoms = len(positions)
    n_molecules = n_atoms // 3
    print(f"Created {n_molecules} water molecules ({n_atoms} atoms)")
    return topology, system, positions


def run_water_box_simulation_with_msd(
    n_waters=216,
    box_size=1.86,
    temperature=300*unit.kelvin,
    timestep=2*unit.femtoseconds,
    n_steps=50000,
    report_interval=100,
    output_dir='water_msd_output'
):
    """
    Run a water box simulation and calculate MSD using AsyncReporter.
    
    Parameters
    ----------
    n_waters : int
        Number of water molecules
    box_size : float
        Box size in nanometers
    temperature : Quantity
        Simulation temperature
    timestep : Quantity
        Integration timestep
    n_steps : int
        Number of simulation steps
    report_interval : int
        How often to calculate MSD
    output_dir : str
        Directory for output files
    """
    print("\n" + "="*70)
    print("Water Box Simulation with Mean Square Displacement")
    print("="*70)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create water box system
    topology, system, positions = create_water_box(n_waters, box_size)
    
    # Create integrator
    friction = 1.0 / unit.picoseconds
    integrator = mm.LangevinMiddleIntegrator(temperature, friction, timestep)
    
    # Create simulation
    simulation = app.Simulation(topology, system, integrator)
    simulation.context.setPositions(positions)
    
    # Minimize, equilibrates
    print("\nMinimizing, equilibrating...")
    simulation.minimizeEnergy(maxIterations=10)
    simulation.step(100)
    # Get initial positions for MSD calculation
    state = simulation.context.getState(getPositions=True)
    initial_positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometers)
    # Add standard reporters
    log_file = str(output_path / 'simulation.log')
    simulation.reporters.append(
        app.StateDataReporter(
            log_file,
            100,
            step=True,
            time=True,
            potentialEnergy=True,
            kineticEnergy=True,
            temperature=True,
            speed=True
        )
    )
    custom_reporter = CustomReporter(
        'log.txt',
        reportInterval=10,
        selection="resid 1"
    )
    simulation.reporters.append(custom_reporter)
    # Run simulation
    print("\nRunning simulation...")
    simulation.step(n_steps)
    exit()
    # Save final structure
    final_state = simulation.context.getState(getPositions=True)
    final_pdb = str(output_path / 'final_structure.pdb')
    with open(final_pdb, 'w') as f:
        app.PDBFile.writeFile(topology, final_state.getPositions(), f)
    
    print(f"\nSimulation complete!")
    print(f"Output directory: {output_dir}/")
    print(f"  • MSD data: msd_data.npz")
    print(f"  • Simulation log: simulation.log")
    print(f"  • Final structure: final_structure.pdb")
    
    return msd_file



# ============================================================================
# Main
# ============================================================================

def main():
    """Run water box simulation with MSD calculation"""
    print("\n" + "="*70)
    print("AsyncReporter Example: Water Box with MSD")
    print("="*70)
    
    # Run the simulation
    run_water_box_simulation_with_msd(
        n_waters=216,           # Small box for quick demo
        box_size=1.86,          # nm
        temperature=300*unit.kelvin,
        timestep=2*unit.femtoseconds,
        n_steps=500,          # 100 ps total
        report_interval=10,    # Calculate MSD every 200 fs
        output_dir='water_msd_output'
    )
    



if __name__ == "__main__":
    main()
