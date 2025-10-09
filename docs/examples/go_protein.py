#!/usr/bin/env python
"""
Simple CG Protein
=================

Simple Go-Martini setup following the workflow pattern

This example demonstrates how to set up a coarse-grained protein system
using the Go-Martini force field, following the same structure as the
production workflows in the workflows directory.

The example includes interactive visualization capabilities using nglview
that can be displayed directly in Jupyter notebooks when building documentation.

For interactive visualization in notebooks::

    # Run the setup
    system = setup_go_protein(sysdir='systems', sysname='test')
    
    # Create visualizations
    viz = visualize_structures(system)
    
    # Display in notebook cells
    viz['original']          # Original atomistic structure
    viz['coarse_grained']    # CG structure  
    viz['system']            # Final solvated system

Requirements:
    - GROMACS
    - Python 3.x
    - reForge package
    - nglview (for visualization): conda install -c conda-forge nglview

Author: DY
"""

from pathlib import Path
from reforge.mdsystem.gmxmd import GmxSystem, GmxRun
from reforge.utils import get_logger
import py3Dmol

logger = get_logger()

# Global settings
INPDB = '1btl.pdb'

def setup_go_protein(sysdir='systems', sysname='test'):
    """Set up a Go-Martini protein system following the workflow pattern.
    
    This function demonstrates the complete setup process for a coarse-grained
    protein system using the Go-Martini force field.
    """
    logger.info(f"Setting up Go-Martini system: {sysname}")
    
    #%%
    # Initialize GmxSystem instance for path management and file handling
    mdsys = GmxSystem(sysdir, sysname)
    inpdb = mdsys.sysdir / INPDB  # Input PDB file location
    print("🔬 Creating original structure visualization...")
    with open(str(inpdb), 'r') as f:
        pdb_data = f.read()
    view_orig = py3Dmol.view(width=800, height=600)
    view_orig.addModel(pdb_data, 'pdb')
    view_orig.setStyle({'cartoon': {'colorscheme': 'chain'}})
    view_orig.setBackgroundColor('white')
    view_orig.zoomTo()
    view_orig.show()
    view_orig.png()
    print("✅ py3Dmol visualization created (PNG export needs Jupyter/browser environment)")
    logger.info("Preparing files and directories...")
    
    # #%%
    # # Prepare necessary files and directories with Martini force field files
    # mdsys.prepare_files(pour_martini=True)
    
    # # List the files in the system's root directory:
    # print("Files in system directory:")
    # for f in mdsys.root.iterdir():
    #     print(f"  {f.name}")

    # #%%
    # # Clean and prepare the input PDB file
    # # Using OpenMM-based cleaning for better compatibility
    # logger.info("Cleaning input PDB file...")
    # mdsys.clean_pdb_mm(inpdb, add_missing_atoms=True, add_hydrogens=True, pH=7.0)
    # print(f"Cleaned PDB: {mdsys.inpdb}")
    
    # # Analyze the original structure
    # print("\n🔬 Analyzing original structure...")
    # try:
    #     from utilities import quick_analysis
    #     stats = quick_analysis(mdsys.inpdb)
    #     print(f"   📊 Structure stats: {stats['num_atoms']} atoms, {stats['num_residues']} residues")
    #     print(f"   � Chains: {', '.join(stats['chains'])}")
    # except Exception as e:
    #     print(f"   ⚠️  Could not analyze original structure: {e}")

    # #%%
    # # Split chains into separate files for processing
    # logger.info("Splitting chains...")
    # mdsys.split_chains()

    # #%%
    # # Coarse-grain the proteins using martinize2 with Go-model parameters
    # logger.info("Applying Go-Martini coarse-graining...")
    # mdsys.martinize_proteins_en(ef=1000, el=0.3, eu=0.9, from_ff='charmm', p="backbone", pf=500, append=False)  # Martini + Elastic network FF 
    # # mdsys.martinize_proteins_go(
    # #     go_eps=12.0,     # Go-model epsilon parameter
    # #     go_low=0.3,      # Lower cutoff for Go contacts
    # #     go_up=1.1,       # Upper cutoff for Go contacts  
    # #     from_ff='amber', # Source force field
    # #     p="backbone",    # Position restraints on backbone
    # #     pf=1000,         # Position restraint force constant
    # #     append=False     # Don't append to existing topology
    # # )

    # #%%
    # # Inspect the generated topology files
    # print("\nGenerated topology files:")
    # for f in mdsys.topdir.iterdir():
    #     print(f"  {f.name}")

    # #%% 
    # # Check the coarse-grained structure files
    # print("\nCoarse-grained structure files:")
    # for f in mdsys.cgdir.iterdir():
    #     print(f"  {f.name}")

    # #%% 
    # # Combine topology and structure files to create system files
    # logger.info("Creating system topology and structure...")
    # mdsys.make_cg_topology()  # Creates system.top
    # mdsys.make_cg_structure()  # Creates solute.pdb
    
    # # Analyze the coarse-grained structure
    # print("\n🔬 Analyzing coarse-grained structure...")
    # try:
    #     from utilities import quick_analysis
    #     if mdsys.solupdb.exists():
    #         stats = quick_analysis(mdsys.solupdb)
    #         print(f"   � CG structure stats: {stats['num_atoms']} beads, {stats['num_residues']} residues")
    #         print(f"   � Chains: {', '.join(stats['chains'])}")
    # except Exception as e:
    #     print(f"   ⚠️  Could not analyze CG structure: {e}")
    
    # # Create simulation box
    # mdsys.make_box(d="1.2", bt="dodecahedron")

    # #%% 
    # # Add solvent and ions to create a realistic simulation environment
    # logger.info("Adding solvent and ions...")
    # solvent = mdsys.root / "water.gro"
    # mdsys.solvate(cp=mdsys.solupdb, cs=solvent, radius="0.17")
    # mdsys.add_bulk_ions(conc=0.10, pname="NA", nname="CL")

    # #%% 
    # # Generate index file for GROMACS selections
    # # Order: 1.System 2.Solute 3.Backbone 4.Solvent 5.Not Water 6+.individual chains
    # logger.info("Creating system index file...")
    # mdsys.make_system_ndx(backbone_atoms=["BB", "BB2"])

    # #%%
    # # Create a visualization-ready structure file
    # logger.info("Creating visualization file...")
    # mdsys.gmx("trjconv", 
    #           clinput='0\n', 
    #           s=mdsys.syspdb, 
    #           f=mdsys.syspdb, 
    #           pbc='atom', 
    #           ur='compact', 
    #           o="viz.pdb")
    
    # logger.info(f"System setup complete for {sysname}")
    # print(f"\nSystem files created in: {mdsys.root}")
    # print(f"Main topology: {mdsys.systop}")
    # print(f"System structure: {mdsys.syspdb}")
    
    # #%%
    # # Demonstrate utilities for file handling and visualization
    # print("\n" + "="*50)
    # print("🛠️  Using reForge Example Utilities")
    # print("="*50)
    
    # # Import our utilities
    # from utilities import list_outputs, quick_analysis
    
    # # List all output files systematically
    # print("\n📋 System Output Summary:")
    # outputs = list_outputs(sysdir, sysname)
    # for category, files in outputs.items():
    #     if files:
    #         print(f"   {category}: {len(files)} files")
    #         for f in files[:3]:  # Show first 3 files
    #             print(f"      - {f}")
    #         if len(files) > 3:
    #             print(f"      ... and {len(files) - 3} more")
    
    # # Analyze the main system PDB
    # if mdsys.syspdb.exists():
    #     print(f"\n🔬 Final System Analysis ({mdsys.syspdb.name}):")
    #     stats = quick_analysis(mdsys.syspdb)
    #     for key, value in stats.items():
    #         if key != "error":
    #             print(f"   {key}: {value}")
    
    # print(f"\n💡 System files ready for MD simulations!")
    # print(f"   📄 Main structure: {mdsys.syspdb}")
    # print(f"   🧬 Main topology: {mdsys.systop}")
    # print(f"   📁 System directory: {mdsys.root}")
    
    # return mdsys


def visualize_structures(mdsys):
    """Create interactive visualizations of the structures using nglview.
    
    This function creates interactive 3D visualizations that can be displayed
    in Jupyter notebooks for documentation purposes.
    
    Args:
        mdsys: GmxSystem instance with structure files
        
    Returns:
        Dictionary with nglview widgets for original and CG structures
    """
    visualizations = {}
    
    try:
        import nglview as nv
        
        # Visualize original structure
        if mdsys.inpdb.exists():
            print("🔬 Creating original structure visualization...")
            view_orig = nv.show_file(str(mdsys.inpdb))
            view_orig.clear_representations()
            view_orig.add_representation('cartoon', color='chain')
            view_orig.camera = 'orthographic'
            view_orig.stage.set_parameters(background_color='white')
            visualizations['original'] = view_orig
            
        # Visualize coarse-grained structure  
        if mdsys.solupdb.exists():
            print("🔬 Creating coarse-grained structure visualization...")
            view_cg = nv.show_file(str(mdsys.solupdb))
            view_cg.clear_representations()
            view_cg.add_representation('ball+stick', color='element')
            view_cg.camera = 'orthographic'
            view_cg.stage.set_parameters(background_color='white')
            visualizations['coarse_grained'] = view_cg
            
        # Visualize final solvated system
        if mdsys.syspdb.exists():
            print("🔬 Creating final system visualization...")
            view_sys = nv.show_file(str(mdsys.syspdb))
            view_sys.clear_representations()
            view_sys.add_representation('cartoon', selection='protein', color='chain')
            view_sys.add_representation('ball+stick', selection='ion', color='element')
            view_sys.camera = 'orthographic'
            view_sys.stage.set_parameters(background_color='white')
            visualizations['system'] = view_sys
            
        print(f"✅ Created {len(visualizations)} interactive visualizations")
        return visualizations
        
    except ImportError:
        print("⚠️  nglview not available. Install with: conda install -c conda-forge nglview")
        return visualizations
    except Exception as e:
        print(f"⚠️  Error creating visualizations: {e}")
        return visualizations


#%%
# Main execution - demonstrate the setup process
if __name__ == "__main__":
    # Set up the Go-Martini protein system
    system = setup_go_protein(sysdir='systems', sysname='test')
    
    # # Create interactive visualizations for documentation
    # print("\n" + "="*50)
    # print("🎨 Creating Interactive Visualizations")
    # print("="*50)
    # visualizations = visualize_structures(system)
    
    # # Display summary of what was created
    # print("\n" + "="*50)
    # print("🎉 Go-Martini Setup Complete!")
    # print("="*50)
    # print(f"System directory: {system.root}")
    # print(f"Ready for MD simulation with GmxRun")
    
    # # Show available visualizations
    # if visualizations:
    #     print(f"\n🔬 Interactive visualizations available:")
    #     for viz_type, widget in visualizations.items():
    #         print(f"   - {viz_type.replace('_', ' ').title()}: Use widget in Jupyter notebook")
    
    # Example of how this would be used in a workflow context:
    # mdrun = GmxRun('systems', 'test', 'run_001')  
    # mdrun.prepare_files()
    # mdrun.empp(f=mdrun.mdpdir / "em_cg.mdp")
    # ... continue with MD simulation steps
