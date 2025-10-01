#!/usr/bin/env python3
"""
Usage: python martinize_rna.py -f ssRNA.pdb -mol rna -elastic yes -ef 100 -el 0.5 -eu 1.2 
-os molecule.pdb -ot molecule.itp

This script processes an all-atom RNA structure and returns coarse-grained topology in the 
GROMACS' .itp format and coarse-grained PDB.
It parses command-line arguments, processes each chain of the input PDB,
maps them to coarse-grained representations, merges the resulting topologies,
optionally applies an elastic network, and writes the output ITP file.
"""

import argparse
import logging
from reforge.forge.forcefields import Martini30RNA
from reforge.forge import cgmap
from reforge.forge.topology import Topology
from reforge.pdbtools import AtomList, pdb2system

# Set up logging - force configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', 
                   datefmt='%H:%M:%S', force=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def martinize_rna(input_pdb, output_topology='molecule.itp', output_structure='molecule.pdb',
                  force_field='reg', molecule_name='molecule', merge_chains='yes',
                  elastic_network='yes', elastic_force=200, elastic_lower=0.3, 
                  elastic_upper=1.2, position_restraints='backbone', 
                  restraint_force=1000, debug=False):
    """
    Main martinization function that can be imported and used programmatically.
    
    Converts an all-atom RNA structure to coarse-grained Martini representation:
    1. Parse all-atom PDB structure
    2. Map to coarse-grained beads using Martini 3.0 RNA force field
    3. Generate topology with bonded interactions
    4. Optionally add elastic network for structural stability
    5. Write coarse-grained structure and topology files
    
    Parameters:
    -----------
    input_pdb : str
        Path to input all-atom RNA structure PDB file
    output_topology : str, optional
        Output topology file path (default: 'molecule.itp')
    output_structure : str, optional
        Output CG structure file path (default: 'molecule.pdb')
    force_field : str, optional
        Force field variant: 'reg' for regular (default: 'reg')
    molecule_name : str, optional
        Molecule name in topology file (default: 'molecule')
    merge_chains : str, optional
        Merge separate chains if detected: 'yes'/'no' (default: 'yes')
    elastic_network : str, optional
        Add elastic network: 'yes'/'no' (default: 'yes')
    elastic_force : float, optional
        Elastic network force constant in kJ/mol/nm² (default: 200)
    elastic_lower : float, optional
        Elastic network lower cutoff in nm (default: 0.3)
    elastic_upper : float, optional
        Elastic network upper cutoff in nm (default: 1.2)
    position_restraints : str, optional
        Position restraints: 'no'/'backbone'/'all' (default: 'backbone')
    restraint_force : float, optional
        Position restraint force constant in kJ/mol/nm² (default: 1000)
    debug : bool, optional
        Enable debug logging (default: False)
        
    Returns:
    --------
    tuple : (str, str)
        Paths to generated structure and topology files
    """
    if debug:
        logger.setLevel(logging.DEBUG)
    
    logger.info("=== Starting RNA Martinization ===")
    logger.info(f"Input PDB: {input_pdb}")
    logger.info(f"Output structure: {output_structure}")
    logger.info(f"Output topology: {output_topology}")
    logger.info(f"Force field: {force_field}")
    logger.info(f"Molecule name: {molecule_name}")
    logger.info(f"Elastic network: {elastic_network}")
    
    if force_field == "reg":
        logger.info("Initializing Martini 3.0 RNA force field")
        ff = Martini30RNA()
    else:
        raise ValueError(f"Unsupported force field option: {force_field}")
    
    logger.info(f"Parsing PDB file: {input_pdb}")
    system = pdb2system(input_pdb)
    logger.info(f"Loaded system with {sum(len(list(chain)) for chain in system.chains())} atoms")
    
    logger.info("Moving O3' atoms to next residues")
    cgmap.move_o3(system)  # Adjust O3 atoms as required
    
    structure = AtomList()
    topologies = []
    start_idx = 1
    chains = list(system.chains())
    logger.info(f"Found {len(chains)} chains to process")
    
    for i, chain in enumerate(chains):
        chain_residues = list(chain)
        sequence_str = " ".join([res.resname for res in chain_residues[:10]])
        if len(chain_residues) > 10:
            sequence_str += "..."
        logger.info(f"Processing chain {i+1}/{len(chains)}")
        logger.info(f"Processing chain with {len(chain_residues)} residues: {sequence_str}")
        
        cg_atoms, chain_top = process_chain(chain, ff, start_idx, molecule_name)
        
        # Log chain topology statistics
        logger.info(f"Chain topology: {len(cg_atoms)} atoms, "
                   f"{len(chain_top.bonds)} bonds, "
                   f"{len(chain_top.angles)} angles, "
                   f"{len(chain_top.dihs)} dihedrals")
        
        structure.extend(cg_atoms)
        topologies.append(chain_top)
        start_idx += len(cg_atoms)
    
    logger.info(f"Total CG atoms generated: {len(structure)}")
    logger.info(f"Writing CG structure to: {output_structure}")
    structure.write_pdb(output_structure)
    
    logger.info(f"Merging {len(topologies)} topology objects")
    merged_topology = merge_topologies(topologies)
    
    # Log merged topology statistics
    logger.info(f"Merged topology: {len(merged_topology.atoms)} atoms, "
               f"{len(merged_topology.bonds)} bonds, "
               f"{len(merged_topology.angles)} angles, "
               f"{len(merged_topology.dihs)} dihedrals")
    
    if elastic_network == 'yes':
        logger.info(f"Adding elastic network (el={elastic_lower}, eu={elastic_upper}, ef={elastic_force})")
        initial_bonds = len(merged_topology.bonds)
        merged_topology.elastic_network(
            structure,
            anames=["BB1", "BB3"],
            el=elastic_lower,
            eu=elastic_upper,
            ef=elastic_force,
        )
        elastic_bonds = len(merged_topology.bonds) - initial_bonds
        logger.info(f"Added {elastic_bonds} elastic bonds")
    
    logger.info(f"Writing topology file to: {output_topology}")
    merged_topology.write_to_itp(output_topology)
    
    logger.info("=== RNA Martinization completed successfully ===")
    logger.info(f"Coarse-grained structure written to: {output_structure}")
    logger.info(f"Topology file written to: {output_topology}")
    
    return output_structure, output_topology


def process_chain(_chain, _ff, _start_idx, _mol_name):
    """
    Process an individual RNA chain: map it to coarse-grained representation and
    generate a topology.

    Args:
        chain (iterable): An RNA chain from the parsed system.
        ff: Force field object.
        start_idx (int): Starting atom index for mapping.
        mol_name (str): Molecule name.

    Returns:
        tuple: (cg_atoms, chain_topology)
    """
    logger.debug(f"Mapping chain to CG representation starting at atom {_start_idx}")
    _cg_atoms = cgmap.map_chain(_chain, _ff, atid=_start_idx)
    
    sequence = [res.resname for res in _chain]
    logger.debug(f"Creating topology for sequence: {' '.join(sequence)}")
    chain_topology = Topology(forcefield=_ff, sequence=sequence, molname=_mol_name)
    
    logger.debug("Processing atoms...")
    chain_topology.process_atoms()
    logger.debug("Processing backbone bonds...")
    chain_topology.process_bb_bonds()
    logger.debug("Processing sidechain bonds...")
    chain_topology.process_sc_bonds()
    
    return _cg_atoms, chain_topology


def merge_topologies(top_list):
    """
    Merge multiple Topology objects into one.

    Args:
        top_list (list): List of Topology objects.

    Returns:
        Topology: The merged Topology.
    """
    _merged_topology = top_list.pop(0)
    for new_top in top_list:
        _merged_topology += new_top
    return _merged_topology


def main():
    """Command-line interface for RNA martinization."""
    parser = argparse.ArgumentParser(
        description="Coarse-grained Martini 3.0 force field for RNA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
WORKFLOW DESCRIPTION:
    This script converts all-atom RNA structures to coarse-grained Martini representation:
    1. Parse all-atom PDB structure and adjust O3' atom positions
    2. Map to coarse-grained beads using Martini 3.0 RNA force field
    3. Generate topology with backbone and sidechain bonded interactions
    4. Optionally add elastic network for enhanced structural stability
    5. Write coarse-grained structure and topology files

USAGE EXAMPLES:
    
    # Basic usage - minimal required arguments
    python martinize_rna.py -f input.pdb
    
    # Full workflow with custom parameters
    python martinize_rna.py -f input.pdb -ot topology.itp -os structure.pdb \\
        -mol my_rna -elastic yes -ef 250 -el 0.25 -eu 1.5
    
    # Using as Python module (programmatic usage):
    >>> from martinize_rna import martinize_rna
    >>> structure, topology = martinize_rna('input.pdb', debug=True)
    >>> print(f"Generated files: {structure}, {topology}")

INPUT REQUIREMENTS:
    - Input PDB: All-atom RNA structure with standard nucleotide naming
    - Structure should have proper chain organization and residue numbering
    - Supported bases: A, U, G, C (standard RNA nucleotides)
        """
    )
    
    parser.add_argument("-f", required=True, type=str, help="Input PDB file")
    parser.add_argument(
        "-ot",
        default="molecule.itp",
        type=str,
        help="Output topology file (default: molecule.itp)",
    )
    parser.add_argument(
        "-os",
        default="molecule.pdb",
        type=str,
        help="Output CG structure (default: molecule.pdb)",
    )
    parser.add_argument(
        "-ff",
        default="reg",
        type=str,
        help="Force field: regular or polar (reg/pol) (default: reg)",
    )
    parser.add_argument(
        "-mol",
        default="molecule",
        type=str,
        help="Molecule name in the .itp file (default: molecule)",
    )
    parser.add_argument(
        "-merge",
        default="yes",
        type=str,
        help="Merge separate chains if detected (default: yes)",
    )
    parser.add_argument(
        "-elastic",
        default="yes",
        type=str,
        help="Add elastic network (default: yes)",
    )
    parser.add_argument(
        "-ef",
        default=200,
        type=float,
        help="Elastic network force constant (default: 200 kJ/mol/nm^2)",
    )
    parser.add_argument(
        "-el",
        default=0.3,
        type=float,
        help="Elastic network lower cutoff (default: 0.3 nm)",
    )
    parser.add_argument(
        "-eu",
        default=1.2,
        type=float,
        help="Elastic network upper cutoff (default: 1.2 nm)",
    )
    parser.add_argument(
        "-p",
        default="backbone",
        type=str,
        help="Output position restraints (no/backbone/all) (default: backbone)",
    )
    parser.add_argument(
        "-pf",
        default=1000,
        type=float,
        help="Position restraints force constant (default: 1000 kJ/mol/nm^2)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    
    args = parser.parse_args()
    
    # Call the main martinization function
    martinize_rna(
        input_pdb=args.f,
        output_topology=args.ot,
        output_structure=args.os,
        force_field=args.ff,
        molecule_name=args.mol,
        merge_chains=args.merge,
        elastic_network=args.elastic,
        elastic_force=args.ef,
        elastic_lower=args.el,
        elastic_upper=args.eu,
        position_restraints=args.p,
        restraint_force=args.pf,
        debug=args.debug
    )


if __name__ == "__main__":
    main()
