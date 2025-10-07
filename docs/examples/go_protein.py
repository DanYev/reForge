#!/usr/bin/env python
"""
Simple CG Protein
=================

Simple Go-Martini setup following the workflow pattern

This example demonstrates how to set up a coarse-grained protein system
using the Go-Martini force field, following the same structure as the
production workflows in the workflows directory.

Requirements:
    - GROMACS
    - Python 3.x
    - reForge package

Author: DY
"""

from pathlib import Path
from reforge.mdsystem.gmxmd import GmxSystem, GmxRun
from reforge.utils import get_logger

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
    inpdb = Path(sysdir) / INPDB  # Input PDB file location
    
    logger.info("Preparing files and directories...")
    
    #%%
    # Prepare necessary files and directories with Martini force field files
    mdsys.prepare_files(pour_martini=True)
    
    # List the files in the system's root directory:
    print("Files in system directory:")
    for f in mdsys.root.iterdir():
        print(f"  {f.name}")

    #%%
    # Clean and prepare the input PDB file
    # Using OpenMM-based cleaning for better compatibility
    logger.info("Cleaning input PDB file...")
    mdsys.clean_pdb_mm(inpdb, add_missing_atoms=True, add_hydrogens=True, pH=7.0)
    print(f"Cleaned PDB: {mdsys.inpdb}")

    #%%
    # Split chains into separate files for processing
    logger.info("Splitting chains...")
    mdsys.split_chains()

    #%%
    # Coarse-grain the proteins using martinize2 with Go-model parameters
    logger.info("Applying Go-Martini coarse-graining...")
    mdsys.martinize_proteins_go(
        go_eps=12.0,     # Go-model epsilon parameter
        go_low=0.3,      # Lower cutoff for Go contacts
        go_up=1.1,       # Upper cutoff for Go contacts  
        from_ff='amber', # Source force field
        p="backbone",    # Position restraints on backbone
        pf=1000,         # Position restraint force constant
        append=False     # Don't append to existing topology
    )

    #%%
    # Inspect the generated topology files
    print("\nGenerated topology files:")
    for f in mdsys.topdir.iterdir():
        print(f"  {f.name}")

    #%% 
    # Check the coarse-grained structure files
    print("\nCoarse-grained structure files:")
    for f in mdsys.cgdir.iterdir():
        print(f"  {f.name}")

    #%% 
    # Combine topology and structure files to create system files
    logger.info("Creating system topology and structure...")
    mdsys.make_cg_topology()  # Creates system.top
    mdsys.make_cg_structure()  # Creates solute.pdb
    
    # Create simulation box
    mdsys.make_box(d="1.2", bt="dodecahedron")

    #%% 
    # Add solvent and ions to create a realistic simulation environment
    logger.info("Adding solvent and ions...")
    solvent = mdsys.root / "water.gro"
    mdsys.solvate(cp=mdsys.solupdb, cs=solvent, radius="0.17")
    mdsys.add_bulk_ions(conc=0.10, pname="NA", nname="CL")

    #%% 
    # Generate index file for GROMACS selections
    # Order: 1.System 2.Solute 3.Backbone 4.Solvent 5.Not Water 6+.individual chains
    logger.info("Creating system index file...")
    mdsys.make_system_ndx(backbone_atoms=["BB", "BB2"])

    #%%
    # Create a visualization-ready structure file
    logger.info("Creating visualization file...")
    mdsys.gmx("trjconv", 
              clinput='0\n', 
              s=mdsys.syspdb, 
              f=mdsys.syspdb, 
              pbc='atom', 
              ur='compact', 
              o="viz.pdb")
    
    logger.info(f"System setup complete for {sysname}")
    print(f"\nSystem files created in: {mdsys.root}")
    print(f"Main topology: {mdsys.systop}")
    print(f"System structure: {mdsys.syspdb}")
    
    #%%
    # Demonstrate utilities for file handling and visualization
    print("\n" + "="*50)
    print("🛠️  Using reForge Example Utilities")
    print("="*50)
    
    # Import our utilities
    from utilities import list_outputs, quick_analysis, create_pdb_html, create_summary_html
    
    # List all output files systematically
    print("\n📋 System Output Summary:")
    outputs = list_outputs(sysdir, sysname)
    
    # Analyze the main system PDB
    if mdsys.syspdb.exists():
        print(f"\n🔬 Quick Analysis of {mdsys.syspdb.name}:")
        stats = quick_analysis(mdsys.syspdb)
        for key, value in stats.items():
            if key != "error":
                print(f"   {key}: {value}")
    
    # Create HTML visualizations
    print("\n🌐 Creating HTML Visualizations...")
    
    # Create visualization for the final system
    if mdsys.syspdb.exists():
        html_viz = create_pdb_html(mdsys.syspdb, 
                                  output_html=mdsys.root / "system_viz.html",
                                  style="cartoon", 
                                  color="chain")
        
    # Create comprehensive system summary
    summary_html = create_summary_html(sysdir, sysname)
    
    print(f"\n💡 Generated HTML files:")
    print(f"   📄 Structure viewer: {mdsys.root}/system_viz.html") 
    print(f"   📊 System summary: {mdsys.root}/{sysname}_summary.html")
    print(f"   🌐 Open these in your browser to explore the system!")
    
    return mdsys


#%%
# Main execution - demonstrate the setup process
if __name__ == "__main__":
    # Set up the Go-Martini protein system
    system = setup_go_protein(sysdir='systems', sysname='test')
    
    # Display summary of what was created
    print("\n" + "="*50)
    print("🎉 Go-Martini Setup Complete!")
    print("="*50)
    print(f"System directory: {system.root}")
    print(f"Ready for MD simulation with GmxRun")
    
    # Example of how this would be used in a workflow context:
    # mdrun = GmxRun('systems', 'test', 'run_001')
    # mdrun.prepare_files()
    # mdrun.empp(f=mdrun.mdpdir / "em_cg.mdp")
    # ... continue with MD simulation steps
