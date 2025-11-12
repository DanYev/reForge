"""
Example usage of AsyncHeavyReporter with OpenMM simulations.
Demonstrates various use cases and heavy calculation patterns.
"""
import numpy as np
from pathlib import Path
import openmm as mm
from openmm import app, unit

# Import the async reporter
from reforge.mdsystem.async_reporter import AsyncHeavyReporter


# ============================================================================
# Example 1: Distance Matrix Calculation
# ============================================================================

def distance_matrix_calculation(data):
    """
    Calculate pairwise distance matrix between all atoms.
    This is computationally expensive for large systems.
    """
    positions = data['positions']
    n_atoms = len(positions)
    
    # Calculate distance matrix (vectorized)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=2))
    
    return {
        'step': data['step'],
        'distance_matrix': distances,
        'mean_distance': np.mean(distances[np.triu_indices(n_atoms, k=1)])
    }


def example_distance_matrix():
    """Example: Track distance matrix evolution during simulation"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Distance Matrix Calculation")
    print("="*70 + "\n")
    
    # This would be your actual OpenMM simulation setup
    # For demonstration, we'll show the reporter setup
    
    reporter = AsyncHeavyReporter(
        'distance_matrices.npz',
        reportInterval=100,  # Report every 100 steps
        calculation_func=distance_matrix_calculation,
        queue_size=20
    )
    
    print("Reporter created for distance matrix calculations")
    print("Add to simulation: simulation.reporters.append(reporter)")
    print("After simulation: reporter.finalize()")
    
    return reporter


# ============================================================================
# Example 2: Contact Map Calculation
# ============================================================================

def contact_map_calculation(data, cutoff=0.5):
    """
    Calculate contact map (atoms within cutoff distance).
    
    Parameters
    ----------
    data : dict
        Frame data with positions
    cutoff : float
        Distance cutoff in nm
    """
    positions = data['positions']
    n_atoms = len(positions)
    
    # Calculate distance matrix
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=2))
    
    # Contact map: 1 if within cutoff, 0 otherwise
    contacts = (distances < cutoff).astype(int)
    
    # Remove self-contacts
    np.fill_diagonal(contacts, 0)
    
    return {
        'step': data['step'],
        'contact_map': contacts,
        'n_contacts': np.sum(contacts) // 2  # Divide by 2 for symmetry
    }


def example_contact_map():
    """Example: Track contact formation/breaking"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Contact Map Calculation")
    print("="*70 + "\n")
    
    reporter = AsyncHeavyReporter(
        'contact_maps.npz',
        reportInterval=500,
        calculation_func=contact_map_calculation,
        cutoff=0.5,  # Additional parameter passed to calculation
        queue_size=15
    )
    
    print("Reporter created for contact map tracking")
    return reporter


# ============================================================================
# Example 3: Radius of Gyration with Subgroups
# ============================================================================

def radius_of_gyration_calculation(data, selection_indices=None):
    """
    Calculate radius of gyration for entire system or selection.
    
    Parameters
    ----------
    data : dict
        Frame data
    selection_indices : list or None
        Indices of atoms to include (None = all atoms)
    """
    positions = data['positions']
    
    if selection_indices is not None:
        positions = positions[selection_indices]
    
    # Center of mass
    com = np.mean(positions, axis=0)
    
    # Radius of gyration
    r_vectors = positions - com
    rg = np.sqrt(np.mean(np.sum(r_vectors**2, axis=1)))
    
    return {
        'step': data['step'],
        'rg': rg,
        'com': com
    }


def example_radius_of_gyration():
    """Example: Track protein compactness"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Radius of Gyration")
    print("="*70 + "\n")
    
    # Assuming CA atoms are indices 0-99
    ca_indices = list(range(0, 100))
    
    reporter = AsyncHeavyReporter(
        'radius_of_gyration.npz',
        reportInterval=100,
        calculation_func=radius_of_gyration_calculation,
        selection_indices=ca_indices,
        queue_size=20
    )
    
    print("Reporter created for Rg tracking (CA atoms only)")
    return reporter


# ============================================================================
# Example 4: Hydrogen Bond Analysis
# ============================================================================

def hydrogen_bond_calculation(data, donor_indices, acceptor_indices, 
                              distance_cutoff=0.35, angle_cutoff=150):
    """
    Detect hydrogen bonds based on geometric criteria.
    
    Parameters
    ----------
    data : dict
        Frame data
    donor_indices : list
        Indices of hydrogen bond donors
    acceptor_indices : list
        Indices of hydrogen bond acceptors
    distance_cutoff : float
        Maximum H...A distance in nm
    angle_cutoff : float
        Minimum D-H...A angle in degrees
    """
    positions = data['positions']
    
    # Simple distance-based detection (full implementation would include angles)
    hbonds = []
    
    for donor_idx in donor_indices:
        for acceptor_idx in acceptor_indices:
            if donor_idx == acceptor_idx:
                continue
            
            distance = np.linalg.norm(positions[donor_idx] - positions[acceptor_idx])
            
            if distance < distance_cutoff:
                hbonds.append((donor_idx, acceptor_idx, distance))
    
    return {
        'step': data['step'],
        'n_hbonds': len(hbonds),
        'hbond_list': hbonds
    }


def example_hydrogen_bonds():
    """Example: Track hydrogen bond dynamics"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Hydrogen Bond Analysis")
    print("="*70 + "\n")
    
    # Example indices (would come from topology)
    donor_indices = [10, 20, 30, 40, 50]
    acceptor_indices = [15, 25, 35, 45, 55]
    
    reporter = AsyncHeavyReporter(
        'hbonds.npz',
        reportInterval=200,
        calculation_func=hydrogen_bond_calculation,
        donor_indices=donor_indices,
        acceptor_indices=acceptor_indices,
        queue_size=25
    )
    
    print("Reporter created for H-bond tracking")
    return reporter


# ============================================================================
# Example 5: Full OpenMM Simulation with Async Reporter
# ============================================================================

def create_simple_system():
    """Create a simple test system for demonstration"""
    # Create a simple system (alanine dipeptide)
    pdb = app.PDBFile('input.pdb')  # Would need actual PDB
    
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    
    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0*unit.nanometer,
        constraints=app.HBonds
    )
    
    integrator = mm.LangevinMiddleIntegrator(
        300*unit.kelvin,
        1.0/unit.picosecond,
        2.0*unit.femtoseconds
    )
    
    simulation = app.Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    
    return simulation


def full_simulation_example():
    """
    Complete example of running OpenMM simulation with async reporter.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Full OpenMM Simulation with AsyncHeavyReporter")
    print("="*70 + "\n")
    
    print("This is a template for a full simulation. Key steps:\n")
    
    code = """
# 1. Create your simulation
simulation = create_simple_system()

# 2. Add standard reporters
simulation.reporters.append(
    app.StateDataReporter('log.txt', 1000, step=True, 
                         potentialEnergy=True, temperature=True)
)

# 3. Add async heavy reporter
heavy_reporter = AsyncHeavyReporter(
    'heavy_analysis.npz',
    reportInterval=1000,
    calculation_func=distance_matrix_calculation,
    queue_size=20
)
simulation.reporters.append(heavy_reporter)

# 4. Run simulation
print("Starting simulation...")
simulation.step(100000)  # MD engine runs at full speed!

# 5. IMPORTANT: Finalize heavy reporter
print("Finalizing heavy calculations...")
heavy_reporter.finalize()  # Wait for all calculations to complete

print("Done! Results saved to heavy_analysis.npz")
"""
    
    print(code)
    print("\nKey points:")
    print("  • MD engine is NOT blocked by heavy calculations")
    print("  • Calculations happen in parallel background thread")
    print("  • Must call finalize() after simulation")
    print("  • Adjust queue_size based on calculation speed")


# ============================================================================
# Example 6: Multiple Reporters
# ============================================================================

def example_multiple_reporters():
    """Example: Use multiple async reporters simultaneously"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Multiple Async Reporters")
    print("="*70 + "\n")
    
    print("You can use multiple async reporters for different analyses:\n")
    
    code = """
# Reporter 1: Distance matrices (expensive)
dist_reporter = AsyncHeavyReporter(
    'distances.npz',
    reportInterval=1000,
    calculation_func=distance_matrix_calculation,
    queue_size=15
)

# Reporter 2: Contact maps (moderate cost)
contact_reporter = AsyncHeavyReporter(
    'contacts.npz',
    reportInterval=500,
    calculation_func=contact_map_calculation,
    queue_size=20
)

# Reporter 3: Radius of gyration (cheap)
rg_reporter = AsyncHeavyReporter(
    'rg.npz',
    reportInterval=100,
    calculation_func=radius_of_gyration_calculation,
    queue_size=30
)

# Add all to simulation
simulation.reporters.extend([dist_reporter, contact_reporter, rg_reporter])

# Run simulation
simulation.step(100000)

# Finalize all
for reporter in [dist_reporter, contact_reporter, rg_reporter]:
    reporter.finalize()
"""
    
    print(code)
    print("\nAll reporters run in parallel - no blocking!")


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("AsyncHeavyReporter - Usage Examples")
    print("="*70)
    
    example_distance_matrix()
    example_contact_map()
    example_radius_of_gyration()
    example_hydrogen_bonds()
    full_simulation_example()
    example_multiple_reporters()
    
    print("\n" + "="*70)
    print("For more information, see:")
    print("  • AsyncHeavyReporter source: reforge/mdsystem/async_reporter.py")
    print("  • Unit tests: tests/test_async_reporter.py")
    print("  • Benchmarks: tests/benchmark_async_reporter.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
