#!/usr/bin/env python3
"""
Backmapping Workflow Script

This script performs a complete coarse-grained to all-atom backmapping workflow:

WORKFLOW:
1. INITIAL BACKMAPPING:
   - Analyze secondary structure of reference AA structure using DSSP
   - Split structure into segments based on secondary structure
   - Segment-wise fitting of reference structure to CG structure positions
   - Generate initial backmapped all-atom structure and optionally add hydrogens to it with --fix-pdb

2. RESTRAINT-BASED MINIMIZATION:
   - Parse mapping files to create atom-to-bead correspondences 
   - Apply position restraints to the heavy atoms (10000 kJ/mol/nm² for CA, 1000 for others)
     with the reference positions from CG beads to maintain overall structure
   - Energy minimize with non-bonded interactions (1.1 nm cutoff)

3. QUALITY ASSESSMENT:
   - Calculate backbone RMSD vs original CG structure
   - Compare secondary structure similarity vs reference structure
   - Report final quality metrics

Usage:
    python backmapping.py -r reference.pdb -s cg.pdb -o final.pdb
"""

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

import mdtraj
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
from pdbfixer import PDBFixer

# Get logger for this module (don't configure it at import time)
logger = logging.getLogger(__name__)

################################################################################
# PRIVATE AUXILIARY FUNCTIONS
################################################################################

# Secondary Structure Analysis Helpers

def _analyze_secondary_structure(pdb_file):
    """Analyze secondary structure of reference structure using MDTraj."""
    try:
        traj = mdtraj.load(pdb_file)
        logger.info("Loaded reference structure: %d residues, %d atoms", traj.n_residues, traj.n_atoms)
        dssp = mdtraj.compute_dssp(traj, simplified=True)
        ss_assignments = dssp[0]
        residue_info = []
        for residue in traj.topology.residues:
            residue_info.append({
                'name': residue.name,
                'index': residue.index,
                'chain_index': residue.chain.index,
                'chain_id': chr(ord('A') + residue.chain.index),
                'ss': ss_assignments[residue.index] if residue.index < len(ss_assignments) else 'C'
            })
        ss_counts = {}
        for ss in ss_assignments:
            ss_counts[ss] = ss_counts.get(ss, 0) + 1
        logger.info("Secondary structure analysis complete:")
        for ss_type, count in ss_counts.items():
            ss_name = {'H': 'α-helix', 'E': 'β-sheet', 'C': 'coil/loop'}.get(ss_type, ss_type)
            logger.info("  %s: %d residues (%.1f%%)", ss_name, count, 100*count/len(ss_assignments))
        return {
            'assignments': ss_assignments,
            'residue_info': residue_info,
            'trajectory': traj,
            'counts': ss_counts
        }
    except Exception as e:
        logger.exception("Failed to analyze secondary structure")
        return None

def _split_into_segments(ss_data, min_segment_length=3):
    """Split structure into continuous secondary structure segments, processing each chain separately."""
    if not ss_data:
        return []
    
    assignments = ss_data['assignments']
    residue_info = ss_data['residue_info']
    
    # Group residues by chain
    chains = {}
    for residue in residue_info:
        chain_id = residue['chain_id']
        if chain_id not in chains:
            chains[chain_id] = []
        chains[chain_id].append(residue)
    
    logger.info(f"Processing {len(chains)} chains separately for segmentation")
    
    all_segments = []
    
    # Process each chain separately
    for chain_id, chain_residues in chains.items():
        logger.debug(f"Segmenting chain {chain_id} with {len(chain_residues)} residues")
        
        segments = []
        current_segment = []
        current_ss_type = None
        
        for i, residue in enumerate(chain_residues):
            # Use the residue's original index to get the correct SS assignment
            ss_type = assignments[residue['index']] if residue['index'] < len(assignments) else 'C'
            
            if current_ss_type is None or ss_type != current_ss_type:
                if current_segment and len(current_segment) >= min_segment_length:
                    segments.append({
                        'ss_type': current_ss_type,
                        'residues': current_segment.copy(),
                        'start_idx': current_segment[0]['index'],
                        'end_idx': current_segment[-1]['index'],
                        'length': len(current_segment),
                        'chain_id': chain_id
                    })
                elif current_segment:
                    if segments:
                        segments[-1]['residues'].extend(current_segment)
                        segments[-1]['end_idx'] = current_segment[-1]['index']
                        segments[-1]['length'] = len(segments[-1]['residues'])
                        segments[-1]['ss_type'] = 'C'
                    else:
                        segments.append({
                            'ss_type': 'C',
                            'residues': current_segment.copy(),
                            'start_idx': current_segment[0]['index'],
                            'end_idx': current_segment[-1]['index'],
                            'length': len(current_segment),
                            'chain_id': chain_id
                        })
                current_segment = [residue]
                current_ss_type = ss_type
            else:
                current_segment.append(residue)
        
        # Handle the last segment for this chain
        if current_segment:
            if len(current_segment) >= min_segment_length:
                segments.append({
                    'ss_type': current_ss_type,
                    'residues': current_segment,
                    'start_idx': current_segment[0]['index'],
                    'end_idx': current_segment[-1]['index'],
                    'length': len(current_segment),
                    'chain_id': chain_id
                })
            elif segments:
                segments[-1]['residues'].extend(current_segment)
                segments[-1]['end_idx'] = current_segment[-1]['index']
                segments[-1]['length'] = len(segments[-1]['residues'])
                segments[-1]['ss_type'] = 'C'
        
        # Add chain segments to all segments
        logger.debug(f"Chain {chain_id}: {len(segments)} segments")
        for segment in segments:
            all_segments.append(segment)
    
    logger.info("Split structure into %d total segments across %d chains", len(all_segments), len(chains))
    for i, segment in enumerate(all_segments):
        ss_name = {'H': 'α-helix', 'E': 'β-sheet', 'C': 'coil/loop'}.get(segment['ss_type'], segment['ss_type'])
        logger.debug("  Segment %d (Chain %s): %s, residues %d-%d (%d residues)", 
                   i+1, segment['chain_id'], ss_name, segment['start_idx'], segment['end_idx'], segment['length'])
    return all_segments

def _extract_ca_positions(trajectory, segment):
    """Extract CA positions for a specific segment from trajectory."""
    ca_positions = []
    ca_indices = []
    for residue_info in segment['residues']:
        residue_idx = residue_info['index']
        for atom in trajectory.topology.atoms:
            if atom.residue.index == residue_idx and atom.name == 'CA':
                ca_indices.append(atom.index)
                break
    if ca_indices:
        ca_positions = trajectory.xyz[0, ca_indices, :]
        ca_positions *= 10.0  # nm to Angstrom
    
    logger.debug(f"Extracted {len(ca_positions)} reference CA positions for segment with {len(segment['residues'])} residues")
    return np.array(ca_positions), ca_indices

def _extract_cg_backbone_positions(cg_traj, cg_residues, segment_start, segment_end):
    """Extract CG backbone positions for corresponding segment."""
    ca_positions = []
    segment_residues = cg_residues[segment_start:segment_end+1]
    logger.debug(f"Extracting CG positions for indices {segment_start}-{segment_end}, "
                f"got {len(segment_residues)} residues")
    
    for i, residue_atoms in enumerate(segment_residues):
        bb_position = None
        for atom in residue_atoms:
            # Only look for BB beads, ignore virtual CA atoms in CG structure
            if atom.name == 'BB':
                # Get position from trajectory using atom index
                bb_position = cg_traj.xyz[0, atom.index] * 10  # Convert nm to Angstroms
                break
        if bb_position is not None:
            ca_positions.append(bb_position)
        else:
            logger.warning(f"No BB backbone bead found for CG residue {segment_start + i}, atoms: {[atom.name for atom in residue_atoms]}")
            # Get positions for all atoms in residue and take mean
            positions = []
            for atom in residue_atoms:
                positions.append(cg_traj.xyz[0, atom.index] * 10)  # Convert nm to Angstroms
            ca_positions.append(np.mean(positions, axis=0))
    
    logger.debug(f"Extracted {len(ca_positions)} CG backbone positions")
    return np.array(ca_positions)

def _fit_segment_to_cg(ref_ca_positions, cg_ca_positions):
    """Fit reference segment CA positions to CG BB positions using optimal rotation and translation."""
    if len(ref_ca_positions) != len(cg_ca_positions):
        logger.error("Mismatch in segment lengths: ref=%d, cg=%d", len(ref_ca_positions), len(cg_ca_positions))
        return None, None
    if len(ref_ca_positions) == 0:
        return np.eye(3), np.zeros(3)
    
    ref_centroid = np.mean(ref_ca_positions, axis=0)
    cg_centroid = np.mean(cg_ca_positions, axis=0)
    ref_centered = ref_ca_positions - ref_centroid
    cg_centered = cg_ca_positions - cg_centroid
    
    if len(ref_ca_positions) == 1:
        rotation_matrix = np.eye(3)
    else:
        H = ref_centered.T @ cg_centered
        U, S, Vt = np.linalg.svd(H)
        rotation_matrix = Vt.T @ U.T
        if np.linalg.det(rotation_matrix) < 0:
            Vt[-1, :] *= -1
            rotation_matrix = Vt.T @ U.T
    
    translation = cg_centroid - rotation_matrix @ ref_centroid
    transformed_ref = (rotation_matrix @ ref_ca_positions.T).T + translation
    rmsd = np.sqrt(np.mean(np.sum((transformed_ref - cg_ca_positions)**2, axis=1)))
    logger.debug("Segment fit RMSD: %.3f Å", rmsd)
    return rotation_matrix, translation

def _apply_transformation_to_segment(ref_trajectory, segment, rotation_matrix, translation):
    """Apply rotation and translation to all atoms in a segment."""
    if rotation_matrix is None or translation is None:
        return None
    transformed_positions = []
    for residue_info in segment['residues']:
        residue_idx = residue_info['index']
        for atom in ref_trajectory.topology.atoms:
            if atom.residue.index == residue_idx:
                orig_pos = ref_trajectory.xyz[0, atom.index, :] * 10.0  # nm to Å
                transformed_pos = (rotation_matrix @ orig_pos) + translation
                transformed_positions.append({
                    'atom': atom,
                    'position': transformed_pos / 10.0  # Å to nm
                })
    return transformed_positions

def _fix_pdb_structure(input_pdb, output_pdb):
    """Fix PDB structure using PDBFixer (optional step)."""
    try:
        logger.info("Fixing PDB structure and adding missing atoms...")
        fixer = PDBFixer(filename=input_pdb)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)  # pH 7.0
        
        with open(output_pdb, 'w') as f:
            app.PDBFile.writeFile(fixer.topology, fixer.positions, f)
        logger.info(f"Fixed structure saved to: {output_pdb}")
        return True
    except Exception as e:
        logger.exception(f"PDB fixing failed: {e}")
        return False

def _create_filtered_temp_cg_pdb(input_cg_pdb):
    """Create a filtered temporary CG PDB file without virtual CA atoms, water, and ions to avoid residue counting issues."""
    temp_fd, temp_cg_pdb = tempfile.mkstemp(suffix='_cg_filtered.pdb', dir=os.path.dirname(input_cg_pdb))
    
    ca_count = 0
    water_ion_count = 0
    total_atoms = 0
    
    # Common water and ion residue names to exclude (including CG-specific names)
    water_ion_names = {
        # Water molecules
        'HOH', 'TIP3', 'TIP4', 'TIP5', 'SPC', 'SPCE', 'WAT', 'H2O', 'W',
        # Common ions (atomistic naming)
        'NA', 'CL', 'K', 'MG', 'CA', 'ZN', 'FE', 'CU', 'MN',
        'NAP', 'CLA', 'POT', 'MG2', 'CA2', 'ZN2', 'FE2', 'FE3',
        'SOD', 'CHL', 'POT', 'MAG', 'CAL',
        # CG-specific ion naming
        'ION', 'Ion'
    }
    
    try:
        with open(input_cg_pdb, 'r') as infile, os.fdopen(temp_fd, 'w') as outfile:
            for line in infile:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    total_atoms += 1
                    
                    # Skip CA atoms in CG structure (they are virtual)
                    if line[12:16].strip() == 'CA':
                        ca_count += 1
                        continue
                    
                    # Skip water and ions
                    residue_name = line[17:20].strip()
                    if residue_name in water_ion_names:
                        water_ion_count += 1
                        continue
                
                # Keep all other lines (including BB, side chains, headers, etc.)
                outfile.write(line)
        
        # Track temporary file for cleanup
        _temp_files.append(temp_cg_pdb)
        excluded_count = ca_count + water_ion_count
        kept_count = total_atoms - excluded_count
        logger.info(f"Created filtered temporary CG PDB excluding {ca_count} virtual CA atoms and {water_ion_count} water/ion atoms (kept {kept_count}/{total_atoms} atoms): {temp_cg_pdb}")
        return temp_cg_pdb
        
    except Exception as e:
        os.close(temp_fd)  # Close file descriptor if still open
        if os.path.exists(temp_cg_pdb):
            os.unlink(temp_cg_pdb)  # Clean up temp file
        logger.error(f"Failed to create temporary CG PDB: {e}")
        raise

def _validate_segments_coverage(segments, ref_trajectory):
    """Validate that segments cover all residues in the structure."""
    try:
        # Get total residues in reference structure
        total_residues = len(list(ref_trajectory.topology.residues))
        
        # Calculate total residues covered by segments
        covered_residues = sum(len(segment['residues']) for segment in segments)
        
        logger.info(f"Segment coverage check: {covered_residues}/{total_residues} residues covered by {len(segments)} segments")
        
        if covered_residues != total_residues:
            logger.error(f"CRITICAL ERROR: Segment coverage mismatch!")
            logger.error(f"  Total residues in structure: {total_residues}")
            logger.error(f"  Residues covered by segments: {covered_residues}")
            logger.error(f"  Missing residues: {total_residues - covered_residues}")
            logger.error("This indicates a bug in segment generation.")
            
            # Show segment details for debugging
            logger.error("Segment details:")
            for i, segment in enumerate(segments):
                logger.error(f"  Segment {i+1} (Chain {segment['chain_id']}): {segment['start_idx']}-{segment['end_idx']} "
                           f"({len(segment['residues'])} residues, {segment['ss_type']})")
            
            return False
        
        logger.info("✅ Segment coverage validation passed.")
        return True
        
    except Exception as e:
        logger.exception(f"Segment validation failed: {e}")
        return False

def _validate_structure_compatibility(ref_trajectory, cg_chains):
    """Validate that reference and CG structures are compatible for backmapping."""
    try:
        # Analyze reference structure chains
        ref_chains = {}
        for residue in ref_trajectory.topology.residues:
            chain_idx = residue.chain.index
            chain_id = chr(ord('A') + chain_idx)
            if chain_id not in ref_chains:
                ref_chains[chain_id] = []
            ref_chains[chain_id].append(residue)
        
        # Log structure information
        logger.info("======== STRUCTURE COMPATIBILITY CHECK ========")
        logger.info("AA reference structure: %d chains, %d total residues", 
                   len(ref_chains), sum(len(residues) for residues in ref_chains.values()))
        for chain_id, residues in ref_chains.items():
            logger.info("  Chain %s: %d residues", chain_id, len(residues))
            first_few = [res.name for res in residues[:5]]
            logger.debug(f"    First 5 residues in chain {chain_id}: {first_few}")
        
        logger.info("CG structure: %d chains, %d total residues", 
                   len(cg_chains), sum(len(residues) for residues in cg_chains.values()))
        for chain_id, residues in cg_chains.items():
            logger.info("  Chain %s: %d residues", chain_id, len(residues))
            first_few = []
            for i, res_atoms in enumerate(residues[:5]):
                if res_atoms:
                    first_few.append(res_atoms[0].residue.name)
            logger.debug(f"    First 5 residues in chain {chain_id}: {first_few}")
        
        # Check for compatibility issues
        errors = []
        warnings = []
        
        # Check if chain count matches
        if len(ref_chains) != len(cg_chains):
            errors.append(f"Chain count mismatch: Reference has {len(ref_chains)} chains, "
                         f"CG structure has {len(cg_chains)} chains")
        
        # Check each chain
        for chain_id in ref_chains:
            if chain_id not in cg_chains:
                errors.append(f"Chain {chain_id} exists in reference but not in CG structure")
                continue
            
            ref_count = len(ref_chains[chain_id])
            cg_count = len(cg_chains[chain_id])
            
            # Check residue count for each chain
            if ref_count != cg_count:
                errors.append(f"Chain {chain_id} residue count mismatch: "
                             f"Reference has {ref_count} residues, CG has {cg_count} residues")
            elif abs(ref_count - cg_count) > 0.1 * max(ref_count, cg_count):  # >10% difference
                warnings.append(f"Chain {chain_id} has significant residue count difference: "
                               f"Reference {ref_count}, CG {cg_count}")
        
        # Check for CG chains not in reference
        for chain_id in cg_chains:
            if chain_id not in ref_chains:
                errors.append(f"Chain {chain_id} exists in CG structure but not in reference")
        
        # Log results
        if errors:
            logger.error("======== STRUCTURE COMPATIBILITY ERRORS ========")
            for error in errors:
                logger.error(f"  ❌ {error}")
        
        if warnings:
            logger.warning("======== STRUCTURE COMPATIBILITY WARNINGS ========")
            for warning in warnings:
                logger.warning(f"  ⚠️  {warning}")
        
        if errors:
            logger.error("Cannot proceed with backmapping due to structural incompatibilities.")
            logger.error("Please ensure that:")
            logger.error("  1. Both structures have the same number of chains")
            logger.error("  2. Each chain has the same number of residues")
            logger.error("  3. Chain IDs match between structures")
            logger.error("  4. CG structure was derived from the same reference structure")
            return False
        
        if warnings:
            logger.warning("Structure compatibility check passed with warnings.")
        else:
            logger.info("✅ Structure compatibility check passed successfully.")
        
        return True
        
    except Exception as e:
        logger.exception(f"Structure compatibility validation failed: {e}")
        return False

def segment_based_backmap(cg_pdb_file, reference_pdb_file, output_pdb_file, min_segment_length=3, temp_cg_pdb=None):
    """Perform segment-based backmapping using secondary structure analysis.
    
    Parameters:
    -----------
    temp_cg_pdb : str
        Pre-created filtered temporary CG PDB without virtual CA atoms, water, and ions.
    """
    try:
        logger.info("Starting segment-based backmapping")
        logger.info("Reference: %s", reference_pdb_file)
        logger.info("CG structure: %s", cg_pdb_file)
        logger.debug("Output: %s", output_pdb_file)
        
        # Use provided filtered temporary CG PDB (always provided from main workflow)
        logger.debug("Using filtered temporary CG PDB (without CA atoms, water, and ions) for analysis: %s", temp_cg_pdb)
        
        # Analyze reference structure secondary structure
        ss_data = _analyze_secondary_structure(reference_pdb_file)
        if not ss_data:
            logger.error("Failed to analyze secondary structure")
            return False
        
        # Split into segments
        segments = _split_into_segments(ss_data, min_segment_length)
        if not segments:
            logger.error("No valid segments found")
            return False
        
        # Validate segments coverage
        if not _validate_segments_coverage(segments, ss_data['trajectory']):
            return False
        
        # Load CG structure and organize by chains (using temp file without CA)
        cg_traj = mdtraj.load(temp_cg_pdb)
        cg_chains = {}
        for residue in cg_traj.topology.residues:
            # Get chain ID from chain index  
            chain_id = chr(ord('A') + residue.chain.index)
            
            if chain_id not in cg_chains:
                cg_chains[chain_id] = []
            # Since we're using filtered temp CG file without CA atoms, water, and ions, include all residues
            cg_chains[chain_id].append(list(residue.atoms))

        logger.info("Loaded CG structure: %d chains, %d total residues", len(cg_chains), cg_traj.n_residues)
        for chain_id, residues in cg_chains.items():
            logger.info("  Chain %s: %d residues", chain_id, len(residues))
            # Debug: show first few residue names
            first_few = []
            for i, res_atoms in enumerate(residues[:5]):
                if res_atoms:
                    first_few.append(res_atoms[0].residue.name)
            logger.debug(f"    First 5 residues in chain {chain_id}: {first_few}")
        
        # Get reference trajectory for processing
        ref_trajectory = ss_data['trajectory']
        
        # Process each segment
        all_transformed_positions = []
        segments_processed = 0
        segments_skipped = 0
        residues_processed = 0
        residues_skipped = 0
        
        for i, segment in enumerate(segments):
            logger.debug("Processing segment %d (Chain %s): %s (%d residues)", 
                       i+1, segment['chain_id'], segment['ss_type'], segment['length'])
            
            # Extract reference CA positions
            ref_ca_positions, _ = _extract_ca_positions(ref_trajectory, segment)
            if len(ref_ca_positions) == 0:
                logger.warning("No CA atoms found for segment %d", i+1)
                continue
            
            # Get chain ID from segment (already determined during segmentation)
            segment_chain_id = segment['chain_id']
            
            if segment_chain_id not in cg_chains:
                logger.warning("No matching CG chain found for segment %d (chain %s)", i+1, segment_chain_id)
                continue
            
            # Extract corresponding CG positions from the matching chain
            cg_residues = cg_chains[segment_chain_id]
            
            # Since structures match, use direct residue mapping
            # Convert global residue indices to chain-local indices (0-based)
            segment_start_local = segment['start_idx']
            segment_end_local = segment['end_idx']
            
            # Find the first residue index for this chain to calculate offset
            first_chain_residue_idx = None
            for residue in ref_trajectory.topology.residues:
                if chr(ord('A') + residue.chain.index) == segment_chain_id:
                    first_chain_residue_idx = residue.index
                    break
            
            if first_chain_residue_idx is None:
                logger.error(f"CRITICAL ERROR: Could not find first residue for chain {segment_chain_id}")
                return False
            
            # Convert to chain-local indices
            segment_start_local = segment['start_idx'] - first_chain_residue_idx
            segment_end_local = segment['end_idx'] - first_chain_residue_idx
            
            # Bounds check
            if segment_start_local < 0 or segment_end_local >= len(cg_residues):
                logger.error(f"CRITICAL ERROR: Segment {i+1} out of bounds!")
                logger.error(f"  Local indices: {segment_start_local}-{segment_end_local}")
                logger.error(f"  CG chain {segment_chain_id} has {len(cg_residues)} residues (0-{len(cg_residues)-1})")
                logger.error("This indicates structure mismatch or indexing bug.")
                return False
            
            # Debug information
            logger.debug(f"Segment {i+1}: global indices {segment['start_idx']}-{segment['end_idx']}, "
                        f"local indices {segment_start_local}-{segment_end_local}, "
                        f"chain {segment_chain_id} (first residue idx {first_chain_residue_idx}), "
                        f"CG chain has {len(cg_residues)} residues")
            
            cg_ca_positions = _extract_cg_backbone_positions(cg_traj, cg_residues, 
                                                            segment_start_local, 
                                                            segment_end_local)
            if len(cg_ca_positions) == 0:
                logger.warning("No CG backbone positions found for segment %d", i+1)
                continue
            
            # Fit segment to CG
            rotation_matrix, translation = _fit_segment_to_cg(ref_ca_positions, cg_ca_positions)
            
            # Apply transformation to all atoms in segment
            transformed_positions = _apply_transformation_to_segment(ref_trajectory, segment, 
                                                                    rotation_matrix, translation)
            if transformed_positions:
                all_transformed_positions.extend(transformed_positions)
        
        # Write output PDB using MDTraj for better compatibility
        if all_transformed_positions:
            # Create new trajectory with transformed positions
            original_traj = ss_data['trajectory']
            n_atoms = len(all_transformed_positions)
            
            # Create position array in nm (MDTraj format)
            new_positions = np.zeros((1, n_atoms, 3))
            
            # Map atom indices to positions
            atom_index_map = {}
            for i, pos_data in enumerate(all_transformed_positions):
                atom = pos_data['atom']
                new_positions[0, i, :] = pos_data['position']  # Already in nm
                atom_index_map[atom.index] = i
            
            # Create a new trajectory by selecting only the transformed atoms
            transformed_atom_indices = [pos_data['atom'].index for pos_data in all_transformed_positions]
            
            # Create new trajectory with original topology but only transformed atoms
            new_traj = original_traj.atom_slice(transformed_atom_indices)
            new_traj.xyz = new_positions
            
            # Save using MDTraj for proper PDB format
            new_traj.save_pdb(output_pdb_file)
            
            logger.info("Backmapping completed: %d atoms written to %s using MDTraj", 
                       len(all_transformed_positions), output_pdb_file)
            return True
        else:
            logger.error("No atoms were transformed")
            return False
    
    except Exception as e:
        logger.exception("Backmapping failed")
        return False


################################################################################
# STEP 2: RESTRAINT-BASED MINIMIZATION CLASSES
################################################################################

class MappingParser:
    """Parse mapping files to understand CG bead to atom correspondences."""
    
    def __init__(self, mapping_dir, force_field):
        self.mapping_dir = Path(mapping_dir)
        self.force_field = force_field
        self.mappings = {}
        
    def load_mapping(self, residue_name):
        """Load mapping for a specific residue."""
        mapping_file = self.mapping_dir / f"{residue_name.lower()}.{self.force_field}.map"
        
        if not mapping_file.exists():
            logger.debug(f"No mapping file found for {residue_name}: {mapping_file}")
            return None
            
        mapping = {}
        try:
            with open(mapping_file, 'r') as f:
                in_atoms_section = False
                for line in f:
                    line = line.strip()
                    if not line or line.startswith(';') or line.startswith('#'):
                        continue
                    
                    if line.startswith('[') and line.endswith(']'):
                        section_name = line[1:-1].strip()
                        in_atoms_section = (section_name == 'atoms')
                        continue
                    
                    if in_atoms_section:
                        parts = line.split()
                        if len(parts) >= 3:
                            atom_name = parts[1]
                            bead_name = parts[2]
                            if bead_name.startswith('!'):
                                continue
                            if bead_name not in mapping:
                                mapping[bead_name] = []
                            mapping[bead_name].append(atom_name)
                            
            logger.debug(f"Loaded mapping for {residue_name}: {len(mapping)} beads")
            return mapping
            
        except Exception as e:
            logger.exception(f"Error loading mapping file {mapping_file}: {e}")
            return None
    
    def get_atom_to_bead_mapping(self, residue_name):
        """Get reverse mapping: atom_name -> bead_name."""
        mapping = self.load_mapping(residue_name)
        if not mapping:
            return {}
        atom_to_bead = {}
        for bead_name, atom_names in mapping.items():
            for atom_name in atom_names:
                atom_to_bead[atom_name] = bead_name
        return atom_to_bead


class RestraintMinimizer:
    """Minimize backmapped structures with CG bead position restraints."""
    
    def __init__(self, aa_structure_file, cg_structure_file, mapping_dir, force_field,
                 output_file, max_iterations=1000, tolerance=10.0, restraint_strength=1000.0, 
                 temp_cg_pdb=None, implicit_solvent=False, fix_pdb_used=False):
        self.aa_structure_file = aa_structure_file
        self.cg_structure_file = cg_structure_file
        self.temp_cg_pdb = temp_cg_pdb  # Pre-created filtered temporary CG PDB without CA atoms, water, and ions
        self.mapping_dir = mapping_dir
        self.force_field = force_field
        self.output_file = output_file
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.restraint_strength = restraint_strength
        self.implicit_solvent = implicit_solvent
        self._fix_pdb_used = fix_pdb_used
        self.aa_structure = None
        self.cg_structure = None
        self.mapping_parser = MappingParser(mapping_dir, force_field)
        self.restraint_positions = {}
        
    def load_structures(self):
        """Load all-atom and coarse-grained structures."""
        self.aa_structure = mdtraj.load(self.aa_structure_file)
        # Use filtered temporary CG structure (without virtual CA atoms, water, and ions) for restraint placement
        self.cg_structure = mdtraj.load(self.temp_cg_pdb)
        logger.info(f"AA structure: {self.aa_structure.n_atoms} atoms, {self.aa_structure.n_residues} residues")
        logger.info(f"CG structure: {self.cg_structure.n_atoms} atoms, {self.cg_structure.n_residues} residues")
    
    def create_restraint_mapping(self):
        """Create mapping from AA atoms to CG bead positions."""
        logger.info("Creating restraint mapping from mapping files...")
        restraint_count = 0
        heavy_atom_count = 0
        chain_restraint_counts = {}
        
        # Group residues by chain for position-based matching
        aa_chains = {}
        cg_chains = {}
        
        for aa_res in self.aa_structure.topology.residues:
            chain_id = aa_res.chain.index
            if chain_id not in aa_chains:
                aa_chains[chain_id] = []
            aa_chains[chain_id].append(aa_res)
        
        for cg_res in self.cg_structure.topology.residues:
            chain_id = cg_res.chain.index
            if chain_id not in cg_chains:
                cg_chains[chain_id] = []
            cg_chains[chain_id].append(cg_res)
        
        logger.info(f"AA chains: {list(aa_chains.keys())}, CG chains: {list(cg_chains.keys())}")
        
        # Match residues by position in chain (sequence order) instead of residue number
        for chain_id in aa_chains:
            if chain_id not in cg_chains:
                logger.warning(f"Chain {chain_id} not found in CG structure")
                continue
            
            aa_chain_residues = aa_chains[chain_id]
            cg_chain_residues = cg_chains[chain_id]
            
            if len(aa_chain_residues) != len(cg_chain_residues):
                logger.warning(f"Chain {chain_id}: Residue count mismatch (AA: {len(aa_chain_residues)}, CG: {len(cg_chain_residues)})")
                # Use the minimum length to avoid index errors
                match_length = min(len(aa_chain_residues), len(cg_chain_residues))
            else:
                match_length = len(aa_chain_residues)
                logger.info(f"Chain {chain_id}: Matching {match_length} residues by sequence position")
            
            # Match residues by their position in the sequence
            for pos in range(match_length):
                aa_res = aa_chain_residues[pos]
                cg_res = cg_chain_residues[pos]
                
                # Log the matching for debugging
                logger.debug(f"Matching position {pos}: AA {aa_res.name}{aa_res.resSeq} <-> CG {cg_res.name}{cg_res.resSeq}")
                
                atom_to_bead = self.mapping_parser.get_atom_to_bead_mapping(aa_res.name)
                if not atom_to_bead:
                    logger.debug(f"No mapping available for residue type {aa_res.name}")
                    continue
                
                cg_bead_positions = {}
                for cg_atom in cg_res.atoms:
                    bead_pos = self.cg_structure.xyz[0, cg_atom.index] * 10
                    cg_bead_positions[cg_atom.name] = bead_pos
                
                for aa_atom in aa_res.atoms:
                    # Skip hydrogen atoms (check both element and atom name starting with H)
                    if aa_atom.element.symbol == 'H' or aa_atom.name.startswith('H'):
                        continue
                        
                    if aa_atom.name in atom_to_bead:
                        bead_name = atom_to_bead[aa_atom.name]
                        
                        if bead_name in cg_bead_positions:
                            self.restraint_positions[aa_atom.index] = {
                                'position': cg_bead_positions[bead_name],
                                'atom_name': aa_atom.name,
                                'residue': aa_res.name,
                                'resid': aa_res.resSeq,
                                'chain': aa_res.chain.index,
                                'bead_name': bead_name
                            }
                            restraint_count += 1
                            heavy_atom_count += 1
                            
                            # Track restraints per chain
                            if chain_id not in chain_restraint_counts:
                                chain_restraint_counts[chain_id] = 0
                            chain_restraint_counts[chain_id] += 1
                            
                            logger.debug(f"Mapped Chain{aa_res.chain.index} {aa_res.name}{aa_res.resSeq}:{aa_atom.name} -> {bead_name}")
        
        logger.info(f"Created {restraint_count} restraints ({heavy_atom_count} heavy atoms)")
        for chain_id, count in chain_restraint_counts.items():
            logger.info(f"  Chain {chain_id}: {count} restraints")
        return restraint_count > 0
    
    def minimize_structure(self):
        """Minimize structure with position restraints."""
        logger.info("Setting up OpenMM minimization...")
        
        # Load PDB file with error handling
        try:
            pdb = app.PDBFile(self.aa_structure_file)
            logger.info(f"Loaded PDB: {pdb.topology.getNumAtoms()} atoms, {pdb.topology.getNumResidues()} residues")
        except Exception as e:
            logger.error(f"Failed to load PDB file {self.aa_structure_file}: {e}")
            logger.error("This could be due to:")
            logger.error("  1. Invalid PDB format from backmapping step")
            logger.error("  2. Missing or malformed atom records")
            logger.error("  3. Unsupported residue types")
            if not self._fix_pdb_used:
                logger.error("Consider using --fix-pdb option to preprocess the structure")
            raise
        
        # Load force field - vacuum is default, implicit solvent optional
        try:
            if self.implicit_solvent:
                forcefield = app.ForceField('amber14-all.xml', 'implicit/gbn2.xml')
                logger.info("Using Amber14 force field with GB/OBC implicit solvent")
            else:
                forcefield = app.ForceField('amber14-all.xml')
                logger.info("Using Amber14 force field in vacuum (default)")
        except Exception as e:
            logger.error(f"Failed to load Amber14 force field: {e}")
            raise
        
        # Create system
        try:
            system = forcefield.createSystem(
                pdb.topology,
                nonbondedMethod=app.CutoffNonPeriodic,
                nonbondedCutoff=1.1 * unit.nanometer,
                constraints=app.HBonds,
                rigidWater=False,
                removeCMMotion=True
            )
            logger.info("System created successfully with H-bond constraints")
        except Exception as e:
            logger.error(f"Failed to create system: {e}")
            logger.error("This could be due to:")
            logger.error("  1. Missing atoms in the structure")
            logger.error("  2. Unrecognized residue types")
            logger.error("  3. Malformed PDB topology")
            if not hasattr(self, '_fix_pdb_used') or not self._fix_pdb_used:
                logger.error("Consider using --fix-pdb option to preprocess the structure")
            raise
        
        logger.info("Using non-bonded interactions with 1.1 nm cutoff")
        
        self.add_position_restraints(system, pdb.topology)
        
        integrator = mm.LangevinMiddleIntegrator(
            300 * unit.kelvin,
            1 / unit.picosecond,
            0.002 * unit.picoseconds
        )
        
        simulation = app.Simulation(pdb.topology, system, integrator)
        simulation.context.setPositions(pdb.positions)
        
        logger.info(f"Starting energy minimization (max {self.max_iterations} iterations)...")
        initial_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
        logger.info(f"Initial potential energy: {initial_energy.value_in_unit(unit.kilojoules_per_mole):.3e} kJ/mol")
        
        simulation.minimizeEnergy(
            tolerance=self.tolerance * unit.kilojoules_per_mole / unit.nanometer,
            maxIterations=self.max_iterations
        )
        
        final_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
        logger.info(f"Final potential energy: {final_energy.value_in_unit(unit.kilojoules_per_mole):.3e} kJ/mol")
        energy_change = final_energy - initial_energy
        logger.info(f"Energy change: {energy_change.value_in_unit(unit.kilojoules_per_mole):.3e} kJ/mol")
        
        state = simulation.context.getState(getPositions=True)
        minimized_positions = state.getPositions()
        
        with open(self.output_file, 'w') as f:
            app.PDBFile.writeFile(pdb.topology, minimized_positions, f)
        
        logger.info(f"Minimized structure saved: {self.output_file}")
        
        return self.output_file
    
    def add_position_restraints(self, system, topology):
        """Add position restraints to the system."""
        logger.info(f"Adding position restraints (CA: 10000, others: {self.restraint_strength} kJ/mol/nm²)...")
        
        restraint_force = mm.CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        restraint_force.addPerParticleParameter("k")
        restraint_force.addPerParticleParameter("x0")
        restraint_force.addPerParticleParameter("y0")
        restraint_force.addPerParticleParameter("z0")
        
        restraint_count = 0
        heavy_atom_restraints = 0
        ca_restraints = 0
        
        for atom in topology.atoms():
            if atom.index in self.restraint_positions:
                restraint_info = self.restraint_positions[atom.index]
                restraint_pos = restraint_info['position']
                
                pos_nm = restraint_pos / 10.0
                
                if atom.name == 'CA':
                    force_constant = 10000.0
                    ca_restraints += 1
                else:
                    force_constant = self.restraint_strength
                
                restraint_force.addParticle(atom.index, [
                    force_constant,
                    pos_nm[0], 
                    pos_nm[1], 
                    pos_nm[2]
                ])
                restraint_count += 1
                heavy_atom_restraints += 1
                
                logger.debug(f"Added restraint: {restraint_info['residue']}{restraint_info['resid']}:"
                           f"{restraint_info['atom_name']} -> {restraint_info['bead_name']}")
        
        if restraint_count > 0:
            system.addForce(restraint_force)
            logger.info(f"Added {restraint_count} position restraints ({ca_restraints} CA atoms @ 10000, "
                       f"{restraint_count - ca_restraints} others @ {self.restraint_strength} kJ/mol/nm²)")
        else:
            logger.warning("No position restraints were added!")
        
        return restraint_force
    
    def run(self):
        """Run the complete restraint-based minimization workflow."""
        try:
            self.load_structures()
            if not self.create_restraint_mapping():
                raise ValueError("No restraint mappings could be created")
            output_file = self.minimize_structure()
            return output_file
        except Exception as e:
            logger.exception(f"Restraint minimization failed: {e}")
            raise


################################################################################
# STEP 3: QUALITY ASSESSMENT FUNCTIONS  
################################################################################

def _calculate_backbone_rmsd(backmapped_pdb, cg_pdb, temp_cg_pdb):
    """Calculate backbone RMSD between backmapped structure and CG structure using MDTraj.
    
    Parameters:
    -----------
    temp_cg_pdb : str
        Pre-created filtered temporary CG PDB without virtual CA atoms, water, and ions.
    """
    try:
        logger.debug("Using filtered temporary CG PDB (without CA atoms, water, and ions) for RMSD calculation: %s", temp_cg_pdb)
        
        backmapped_traj = mdtraj.load(backmapped_pdb)
        cg_traj = mdtraj.load(temp_cg_pdb)
        
        # Get chain information
        backmapped_chains = {}
        # Select CA atoms from backmapped structure
        ca_indices = backmapped_traj.topology.select("name CA")
        ca_positions = backmapped_traj.xyz[0, ca_indices, :] * 10  # nm to Angstrom
        
        ca_atom_idx = 0
        for atom in backmapped_traj.topology.atoms:
            if atom.name == 'CA':
                chain_id = chr(ord('A') + atom.residue.chain.index)
                if chain_id not in backmapped_chains:
                    backmapped_chains[chain_id] = []
                backmapped_chains[chain_id].append(ca_positions[ca_atom_idx])
                ca_atom_idx += 1
        
        cg_chains = {}
        # Only select BB beads from CG structure (ignore virtual CA atoms)
        bb_indices = cg_traj.topology.select("name BB")
        bb_positions = cg_traj.xyz[0, bb_indices, :] * 10  # nm to Angstrom
        
        bb_atom_idx = 0
        for atom in cg_traj.topology.atoms:
            if atom.name == 'BB':
                chain_id = chr(ord('A') + atom.residue.chain.index)
                if chain_id not in cg_chains:
                    cg_chains[chain_id] = []
                cg_chains[chain_id].append(bb_positions[bb_atom_idx])
                bb_atom_idx += 1
        
        if not backmapped_chains or not cg_chains:
            logger.warning("No backbone atoms found in one or both structures")
            return None
        
        # Calculate RMSD for matching chains
        all_squared_distances = []
        total_atoms = 0
        
        for chain_id in backmapped_chains:
            if chain_id in cg_chains:
                backmapped_pos = np.array(backmapped_chains[chain_id])
                cg_pos = np.array(cg_chains[chain_id])
                
                # Use minimum length for this chain
                min_len = min(len(backmapped_pos), len(cg_pos))
                if min_len > 0:
                    backmapped_pos = backmapped_pos[:min_len]
                    cg_pos = cg_pos[:min_len]
                    
                    # Calculate squared distances for this chain
                    squared_distances = np.sum((backmapped_pos - cg_pos)**2, axis=1)
                    all_squared_distances.extend(squared_distances)
                    total_atoms += min_len
                    
                    logger.debug(f"Chain {chain_id}: {min_len} atoms, RMSD = {np.sqrt(np.mean(squared_distances)):.3f} Å")
        
        if len(all_squared_distances) == 0:
            logger.warning("No matching backbone positions found between chains")
            return None
        
        # Calculate overall RMSD
        rmsd = np.sqrt(np.mean(all_squared_distances))
        logger.debug(f"Overall backbone RMSD: {rmsd:.3f} Å from {total_atoms} atoms")
        
        return rmsd
    except Exception as e:
        logger.exception("Failed to calculate backbone RMSD")
        return None

def _calculate_ss_similarity(backmapped_pdb, reference_pdb):
    """Calculate secondary structure similarity between backmapped and reference structures."""
    try:
        backmapped_traj = mdtraj.load(backmapped_pdb)
        backmapped_dssp = mdtraj.compute_dssp(backmapped_traj, simplified=True)[0]
        reference_traj = mdtraj.load(reference_pdb)
        reference_dssp = mdtraj.compute_dssp(reference_traj, simplified=True)[0]
        min_length = min(len(backmapped_dssp), len(reference_dssp))
        if min_length == 0:
            logger.warning("No residues to compare for SS similarity")
            return None
        backmapped_dssp = backmapped_dssp[:min_length]
        reference_dssp = reference_dssp[:min_length]
        matches = np.sum(backmapped_dssp == reference_dssp)
        similarity = matches / min_length
        logger.info("Secondary structure comparison:")
        for ss_type in ['H', 'E', 'C']:
            ss_name = {'H': 'α-helix', 'E': 'β-sheet', 'C': 'coil/loop'}[ss_type]
            ref_count = np.sum(reference_dssp == ss_type)
            back_count = np.sum(backmapped_dssp == ss_type)
            preserved = np.sum((reference_dssp == ss_type) & (backmapped_dssp == ss_type))
            preservation_rate = preserved / ref_count if ref_count > 0 else 0
            logger.info("  %s: Reference %d (%.1f%%), Backmapped %d (%.1f%%), Preserved %d (%.1f%%)",
                       ss_name, ref_count, 100*ref_count/min_length, 
                       back_count, 100*back_count/min_length,
                       preserved, 100*preservation_rate)
        return similarity
    except Exception as e:
        logger.exception("Failed to calculate SS similarity")
        return None


def cleanup_temp_files(temp_files):
    """Clean up all tracked temporary files."""
    cleaned_count = 0
    for temp_file in temp_files[:]:  # Create a copy to iterate over
        if os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
                temp_files.remove(temp_file)
                logger.debug(f"Cleaned up temporary file: {temp_file}")
                cleaned_count += 1
            except OSError as e:
                logger.warning(f"Could not remove temporary file {temp_file}: {e}")
    return cleaned_count   

################################################################################
# MAIN WORKFLOW FUNCTION
################################################################################

def backmap(cg_structure_pdb, aa_reference_pdb, output_pdb='backmapped.pdb', 
           mapping_dir='_map_files_v3', force_field='amber', 
           min_segment_length=3, max_iterations=1000, tolerance=10.0, 
           restraint_strength=1000.0, debug=False, fix_pdb=False, implicit_solvent=False):
    """
    Main backmapping function that can be imported and used programmatically.
    
    Performs coarse-grained to all-atom backmapping with three main steps:
    1. Initial segment-based backmapping from CG to AA
    2. Restraint-based minimization with position restraints  
    3. Backbone RMSD calculation vs CG structure and SS comparison vs reference
    
    Parameters:
    -----------
    cg_structure_pdb : str
        Path to coarse-grained structure PDB file
    aa_reference_pdb : str
        Path to reference all-atom topology PDB file
    output_pdb : str, optional
        Final output filename (default: 'backmapped.pdb')
    mapping_dir : str, optional
        Directory containing mapping files (default: '_map_files_v3')
    force_field : str, optional
        Force field to use (default: 'amber')
    min_segment_length : int, optional
        Minimum segment length for SS-based segmentation (default: 3)
    max_iterations : int, optional
        Maximum minimization iterations (default: 1000)
    tolerance : float, optional
        Energy tolerance for minimization (default: 10.0)
    restraint_strength : float, optional
        Restraint force constant for non-CA atoms (default: 1000.0)
    debug : bool, optional
        Enable debug logging and keep intermediate files (default: False)
    fix_pdb : bool, optional
        Apply PDBFixer to initial structure (default: False)
    implicit_solvent : bool, optional
        Use implicit solvent (GB/OBC) instead of vacuum (default: False)
        
    Returns:
    --------
    str
        Path to the final backmapped PDB file
        
    Raises:
    -------
    SystemExit
        If any step in the backmapping workflow fails
        
    Examples:
    ---------
    >>> # Basic usage
    >>> output_file = backmap('cg_structure.pdb', 'reference.pdb')
    
    >>> # With custom parameters
    >>> output_file = backmap(
    ...     cg_structure_pdb='my_cg.pdb',
    ...     aa_reference_pdb='my_ref.pdb', 
    ...     output_pdb='final_structure.pdb',
    ...     debug=True,
    ...     max_iterations=2000
    ... )
    """
    # Initialize temporary files list for this workflow
    global _temp_files
    _temp_files = []
    
    # Set up logging only if it hasn't been configured yet (no handlers exist)
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        # Configure basic logging for when backmap() is called programmatically
        # Always set console to INFO level
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
    
    if debug:
        # Set up dual logging: DEBUG to file, INFO to console
        logger.setLevel(logging.DEBUG)
        
        # Ensure console handler stays at INFO level
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                handler.setLevel(logging.INFO)
        
        # Only add file handler if we don't already have one
        has_file_handler = any(isinstance(h, logging.FileHandler) for h in root_logger.handlers)
        if not has_file_handler:
            # File handler (DEBUG level) - only create when debugging
            file_handler = logging.FileHandler('tmp_debug_bm.log', mode='w')
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
        
        logger.info("Debug mode enabled: detailed logging saved to tmp_debug_bm.log")
    else:
        # Normal mode: only console logging at INFO level
        logger.setLevel(logging.INFO)
    
    # temp_cg_pdb will be tracked in _temp_files
    temp_cg_pdb = None
    
    try:
        # Create filtered temporary CG PDB without virtual CA atoms, water, and ions once for the entire workflow
        logger.info("======== PREPROCESSING: Creating filtered temporary CG structure ========")
        logger.info("Removing virtual CA atoms, water molecules, and ions from CG structure for consistent residue counting")
        temp_cg_pdb = _create_filtered_temp_cg_pdb(cg_structure_pdb)
        
        # Load structures for compatibility check
        ref_traj = mdtraj.load(aa_reference_pdb)
        cg_traj = mdtraj.load(temp_cg_pdb)
        
        # Organize CG chains for compatibility check
        cg_chains = {}
        for residue in cg_traj.topology.residues:
            chain_id = chr(ord('A') + residue.chain.index)
            if chain_id not in cg_chains:
                cg_chains[chain_id] = []
            cg_chains[chain_id].append(list(residue.atoms))
        
        # Check compatibility before proceeding
        if not _validate_structure_compatibility(ref_traj, cg_chains):
            logger.error("Structure compatibility check failed - aborting workflow")
            sys.exit(1)
        
        # Step 1: Initial backmapping
        logger.info("======== STEP 1: Initial Backmapping ========")
        # Create temporary file for initial backmapping output in CG structure directory
        cg_dir = os.path.dirname(cg_structure_pdb)
        temp_fd, intermediate_file = tempfile.mkstemp(suffix='_backmapped_initial.pdb', dir=cg_dir)
        os.close(temp_fd)  # Close file descriptor, we just need the filename
        _temp_files.append(intermediate_file)
        
        success = segment_based_backmap(
            cg_pdb_file=cg_structure_pdb,
            reference_pdb_file=aa_reference_pdb,
            output_pdb_file=intermediate_file,
            min_segment_length=min_segment_length,
            temp_cg_pdb=temp_cg_pdb
        )
        
        if not success:
            logger.error("Initial backmapping failed")
            sys.exit(1)
        
        # Optional: Fix PDB structure if requested
        minimization_input = intermediate_file
        fixed_file = None
        if fix_pdb:
            logger.info("======== OPTIONAL: PDB Structure Fixing ========")
            # Create temporary file for fixed structure in CG structure directory
            temp_fd, fixed_file = tempfile.mkstemp(suffix='_backmapped_fixed.pdb', dir=cg_dir)
            os.close(temp_fd)  # Close file descriptor, we just need the filename
            _temp_files.append(fixed_file)
            
            if _fix_pdb_structure(intermediate_file, fixed_file):
                minimization_input = fixed_file
                logger.info("Using fixed structure for minimization")
            else:
                logger.warning("PDB fixing failed, using original backmapped structure")
                # Remove from temp files list if it wasn't created
                if fixed_file in _temp_files:
                    _temp_files.remove(fixed_file)
        
        # Step 2: Minimization with restraints
        logger.info("======== STEP 2: Restraint-based Minimization ========")
        logger.info("Using CG structure for restraint placement at CG bead positions")
        minimizer = RestraintMinimizer(
            aa_structure_file=minimization_input,
            cg_structure_file=cg_structure_pdb,
            mapping_dir=mapping_dir,
            force_field=force_field,
            output_file=output_pdb,
            max_iterations=max_iterations,
            tolerance=tolerance,
            restraint_strength=restraint_strength,
            temp_cg_pdb=temp_cg_pdb,
            implicit_solvent=implicit_solvent,
            fix_pdb_used=fix_pdb
        )
        
        final_output = minimizer.run()

        # Step 3: Backmapped AA vs CG Comparison
        logger.info("======== STEP 3: Backmapped AA vs CG Comparison ========")

        backbone_rmsd = _calculate_backbone_rmsd(final_output, cg_structure_pdb, temp_cg_pdb)
        if backbone_rmsd is not None:
            logger.info(f"Backbone RMSD vs CG structure: {backbone_rmsd:.3f} Å")

        ss_similarity = _calculate_ss_similarity(final_output, aa_reference_pdb)
        if ss_similarity is not None:
            logger.info(f"Overall SS similarity: {ss_similarity:.3f} ({ss_similarity*100:.1f}%)")
        
        logger.info(f"✅ Complete workflow finished successfully: {final_output}")
        return final_output
        
    except Exception as e:
        logger.exception(f"❌ Workflow failed: {e}")
        sys.exit(1)
    
    finally:
        # Clean up all temporary files (unless debug is specified)
        if debug:
            logger.info("======== CLEANUP: Keeping Temporary Files (--debug mode) ========")
            existing_temp = [f for f in _temp_files if os.path.exists(f)]
            
            if existing_temp:
                logger.info(f"Temporary files preserved: {', '.join(existing_temp)}")
            logger.info("Debug log saved to: tmp_debug_bm.log")
        else:
            logger.debug("======== CLEANUP: Removing Temporary Files ========")
            
            # Clean up all temporary files using the existing function
            cleaned_temp_count = cleanup_temp_files(_temp_files)
            
            # Report cleanup results
            if cleaned_temp_count > 0:
                logger.debug(f"Cleaned up {cleaned_temp_count} temporary files")
            else:
                logger.debug("No files to clean up")


def main():
    """Command-line interface for the backmapping workflow."""
    parser = argparse.ArgumentParser(
        description="Combined backmapping, minimization, RMSD and SS comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
WORKFLOW DESCRIPTION:
    This script performs complete coarse-grained to all-atom backmapping:
    1. Initial segment-based backmapping from CG to AA structure
    2. Restraint-based minimization with position restraints on non-CA atoms
    3. Backbone RMSD calculation vs original CG structure
    4. Secondary structure comparison vs reference all-atom structure

USAGE EXAMPLES:
    
    # Basic usage - minimal required arguments
    python backmapping.py -r reference.pdb -c cg_structure.pdb
    
    # Full workflow with custom parameters and output
    python backmapping.py -r reference.pdb -c cg_structure.pdb -o final.pdb \\
        --force-field amber --max-iterations 2000 --debug
    
    # Using as Python module (programmatic usage):
    >>> from backmapping import backmap
    >>> result = backmap('cg_structure.pdb', 'reference.pdb', debug=True)
    >>> print(f"Backmapped structure saved to: {result}")

INPUT REQUIREMENTS:
    - Reference PDB: All-atom structure for topology and secondary structure comparison
    - CG Structure PDB: Coarse-grained structure to be backmapped to all-atom
    - Both files should have matching sequences and chain organization
        """
    )
    
    parser.add_argument('-r', '--aa-reference', required=True,
                       help='Reference all-atom topology PDB file')
    parser.add_argument('-s', '--cg-structure', required=True,
                       help='Coarse-grained structure PDB file')
    parser.add_argument('-o', '--output', default='backmapped.pdb',
                       help='Final output filename (default: backmapped.pdb)')
    parser.add_argument('-m', '--mapping-dir', default='_map_files_v3',
                       help='Directory containing mapping files (default: _map_files_v3)')
    parser.add_argument('-f', '--force-field', default='amber',
                       help='Force field: amber, charmm36, etc. (default: amber)')
    parser.add_argument('--min-segment-length', type=int, default=3,
                       help='Minimum segment length for SS-based segmentation (default: 3)')
    parser.add_argument('--max-iterations', type=int, default=1000,
                       help='Maximum minimization iterations (default: 1000)')
    parser.add_argument('--tolerance', type=float, default=10.0,
                       help='Energy tolerance for minimization in kJ/mol (default: 10.0)')
    parser.add_argument('--restraint-strength', type=float, default=1000.0,
                       help='Restraint force constant for non-CA atoms in kJ/mol/nm² (default: 1000.0)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging to debug_bm.log and keep intermediate files')
    parser.add_argument('--fix-pdb', action='store_true',
                       help='Apply PDBFixer to initial backmapped structure (adds missing atoms/hydrogens)')
    parser.add_argument('--implicit-solvent', action='store_true',
                       help='Use implicit solvent (GB/OBC) instead of vacuum for minimization (default: vacuum)')
    
    args = parser.parse_args()
    
    # Set up logging for command-line usage
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Call the main backmapping function with parsed arguments
    backmap(
        cg_structure_pdb=args.cg_structure,
        aa_reference_pdb=args.aa_reference,
        output_pdb=args.output,
        mapping_dir=args.mapping_dir,
        force_field=args.force_field,
        min_segment_length=args.min_segment_length,
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
        restraint_strength=args.restraint_strength,
        debug=args.debug,
        fix_pdb=args.fix_pdb,
        implicit_solvent=args.implicit_solvent
    )


if __name__ == "__main__":
    main()
