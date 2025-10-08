#!/usr/bin/env python
"""
Utilities for visualization and analysis in reForge examples.

This module provides helper functions for creating visualizations
and analyzing molecular dynamics systems, specifically designed
for use in documentation examples.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import tempfile

def list_outputs(sysdir: str, sysname: str) -> Dict[str, List[str]]:
    """List all output files from a system setup systematically.
    
    Args:
        sysdir: System directory name
        sysname: System name
        
    Returns:
        Dictionary of file categories and their files
    """
    system_path = Path(sysdir) / sysname
    outputs = {
        'structure_files': [],
        'topology_files': [],
        'visualization_files': [],
        'other_files': []
    }
    
    if not system_path.exists():
        return outputs
    
    for file_path in system_path.rglob("*"):
        if file_path.is_file():
            ext = file_path.suffix.lower()
            rel_path = str(file_path.relative_to(system_path))
            
            if ext in ['.pdb', '.gro', '.mol2', '.xyz']:
                outputs['structure_files'].append(rel_path)
            elif ext in ['.top', '.itp', '.prm', '.psf']:
                outputs['topology_files'].append(rel_path)
            elif ext in ['.html', '.png', '.jpg', '.jpeg', '.svg']:
                outputs['visualization_files'].append(rel_path)
            else:
                outputs['other_files'].append(rel_path)
    
    # Sort all lists
    for key in outputs:
        outputs[key].sort()
    
    return outputs


def quick_analysis(pdb_path: Path) -> Dict[str, Any]:
    """Perform quick analysis of a PDB file.
    
    Args:
        pdb_path: Path to PDB file
        
    Returns:
        Dictionary with structure statistics
    """
    stats = {
        'file_size': 'N/A',
        'num_atoms': 0,
        'num_residues': 0,
        'chains': [],
        'error': None
    }
    
    try:
        if pdb_path.exists():
            stats['file_size'] = f"{pdb_path.stat().st_size / 1024:.1f} KB"
            
            atoms = set()
            residues = set()
            chains = set()
            
            with open(pdb_path, 'r') as f:
                for line in f:
                    if line.startswith(('ATOM', 'HETATM')):
                        chain = line[21:22].strip()
                        res_num = line[22:26].strip()
                        res_name = line[17:20].strip()
                        
                        atoms.add(line[12:16].strip())
                        residues.add(f"{chain}_{res_name}_{res_num}")
                        if chain:
                            chains.add(chain)
            
            stats['num_atoms'] = len(atoms)
            stats['num_residues'] = len(residues)
            stats['chains'] = sorted(list(chains)) if chains else ['A']
            
    except Exception as e:
        stats['error'] = str(e)
    
    return stats









