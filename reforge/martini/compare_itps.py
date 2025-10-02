#!/usr/bin/env python3
"""
Compare two ITP files section by section with detailed atom-by-atom and bond-by-bond analysis.
"""

import sys
import re
from collections import defaultdict

def get_section_description(section_name):
    """Get a description of what each topology section contains."""
    descriptions = {
        'defaults': 'Default parameters (nbfunc, comb-rule, gen-pairs, fudgeLJ, fudgeQQ)',
        'moleculetype': 'Molecule type definition (name, nrexcl)',
        'atoms': 'Atom definitions (id, type, resid, resname, atomname, cgnr, charge, mass)',
        'bonds': 'Bonded interactions between pairs of atoms',
        'pairs': 'Non-bonded pairs with special interactions (1-4 interactions)',
        'angles': 'Angular interactions between triplets of atoms',
        'dihedrals': 'Dihedral angle interactions between quadruplets of atoms',
        'constraints': 'Distance constraints between atoms',
        'settles': 'SETTLE algorithm constraints for water molecules',
        'exclusions': 'Excluded non-bonded interactions',
        'position_restraints': 'Position restraints for atoms',
        'virtual_sites2': 'Virtual sites constructed from 2 atoms',
        'virtual_sites3': 'Virtual sites constructed from 3 atoms',
        'virtual_sites4': 'Virtual sites constructed from 4 atoms',
        'virtual_sitesn': 'Virtual sites constructed from N atoms',
        'system': 'System name',
        'molecules': 'Molecules in the system and their counts',
        'atomtypes': 'Atom type definitions',
        'bondtypes': 'Bond type parameters',
        'angletypes': 'Angle type parameters',
        'dihedraltypes': 'Dihedral type parameters',
        'constrainttypes': 'Constraint type parameters',
        'nonbond_params': 'Non-bonded interaction parameters',
        'pairtypes': 'Pair interaction parameters'
    }
    return descriptions.get(section_name, 'Unknown section type')

def parse_itp_sections(filename):
    """Parse an ITP file and return a dictionary of sections with their content."""
    sections = {}
    current_section = None
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith(';'):
                    continue
                
                # Check for section headers
                if line.startswith('[') and line.endswith(']'):
                    current_section = line[1:-1].strip()
                    sections[current_section] = []
                elif current_section and not line.startswith('#'):
                    # Add non-comment lines to current section
                    sections[current_section].append(line)
                    
    except FileNotFoundError:
        print(f"Error: Could not find file {filename}")
        return {}
    
    return sections

def compare_itp_files(file1, file2):
    """Compare two ITP files section by section."""
    print(f"Comparing {file1} vs {file2}")
    print("=" * 60)
    
    sections1 = parse_itp_sections(file1)
    sections2 = parse_itp_sections(file2)
    
    print(f"Found {len(sections1)} sections in {file1}: {list(sections1.keys())}")
    print(f"Found {len(sections2)} sections in {file2}: {list(sections2.keys())}")
    print()
    
    # Get all section names from both files
    all_sections = set(sections1.keys()) | set(sections2.keys())
    
    # Define the standard order for topology sections
    topology_sections_order = [
        "defaults", "moleculetype", "atoms", "bonds", "pairs", "angles", 
        "dihedrals", "constraints", "settles", "exclusions", "position_restraints",
        "virtual_sites2", "virtual_sites3", "virtual_sites4", "virtual_sitesn",
        "system", "molecules", "atomtypes", "bondtypes", "angletypes", 
        "dihedraltypes", "constrainttypes", "nonbond_params", "pairtypes"
    ]
    
    # Sort sections: first by standard order, then alphabetically for any others
    def section_sort_key(section):
        if section in topology_sections_order:
            return (0, topology_sections_order.index(section))
        else:
            return (1, section)
    
    sorted_sections = sorted(all_sections, key=section_sort_key)
    
    print("=== Section-by-Section Comparison ===")
    # Compare each section
    for section in sorted_sections:
        count1 = len(sections1.get(section, []))
        count2 = len(sections2.get(section, []))
        
        status = "✓" if count1 == count2 else "✗"
        
        print(f"{status} [{section:20s}] {file1}: {count1:4d} | {file2}: {count2:4d}")
        
        # Show difference if counts don't match
        if count1 != count2:
            print(f"  → Difference: {count2 - count1:+d}")
    
    print("=" * 60)
    
    # Summary
    total1 = sum(len(content) for content in sections1.values())
    total2 = sum(len(content) for content in sections2.values())
    
    print(f"Total entries: {file1}: {total1} | {file2}: {total2}")
    if total1 == total2:
        print("✓ Files have the same total number of entries!")
    else:
        print(f"✗ Difference in total entries: {total2 - total1:+d}")

def parse_interaction_line(line):
    """Parse a bonded interaction line and return atom indices and parameters."""
    parts = line.split(';')[0].strip().split()  # Remove comments and split
    if len(parts) < 2:
        return None
    
    # First parts are atom indices, then function type, then parameters
    try:
        atom_indices = []
        params = []
        func_type = None
        
        # Find where atom indices end (first non-integer after indices)
        for i, part in enumerate(parts):
            try:
                val = int(part)
                if i == 0 or (i > 0 and isinstance(atom_indices[-1], int)):
                    atom_indices.append(val)
                else:
                    # This is the function type
                    func_type = val
                    params = parts[i+1:]
                    break
            except ValueError:
                # This is a parameter (float or string)
                if func_type is None and len(atom_indices) > 0:
                    # Assume previous integer was function type
                    func_type = atom_indices.pop()
                params = parts[i:]
                break
        
        return {
            'atoms': tuple(atom_indices),
            'func_type': func_type,
            'params': params,
            'raw': line.strip()
        }
    except (ValueError, IndexError):
        return {'raw': line.strip(), 'atoms': (), 'func_type': None, 'params': []}

def parse_atom_line(line):
    """Parse an atoms section line."""
    parts = line.split(';')[0].strip().split()  # Remove comments
    if len(parts) < 8:
        return {'raw': line.strip()}
    
    try:
        return {
            'id': int(parts[0]),
            'type': parts[1], 
            'resnr': int(parts[2]),
            'resname': parts[3],
            'atomname': parts[4],
            'cgnr': int(parts[5]),
            'charge': float(parts[6]),
            'mass': float(parts[7]),
            'raw': line.strip()
        }
    except (ValueError, IndexError):
        return {'raw': line.strip()}

def compare_section_detailed(sections1, sections2, section_name, file1, file2):
    """Compare a section in detail, showing differences."""
    print(f"\n=== Detailed comparison of [{section_name}] section ===")
    
    data1 = sections1.get(section_name, [])
    data2 = sections2.get(section_name, [])
    
    print(f"Section [{section_name}] - {file1}: {len(data1)} entries, {file2}: {len(data2)} entries")
    
    if not data1 and not data2:
        print(f"Both files have empty [{section_name}] sections")
        return
    elif not data1:
        print(f"Section [{section_name}] not found in {file1}, but has {len(data2)} entries in {file2}")
        print("First few entries from", file2)
        for i, line in enumerate(data2[:5]):
            print(f"  {i+1:3d}: {line}")
        return
    elif not data2:
        print(f"Section [{section_name}] not found in {file2}, but has {len(data1)} entries in {file1}")
        print("First few entries from", file1)
        for i, line in enumerate(data1[:5]):
            print(f"  {i+1:3d}: {line}")
        return
    
    # Parse section data based on type
    if section_name == 'atoms':
        parsed1 = [parse_atom_line(line) for line in data1]
        parsed2 = [parse_atom_line(line) for line in data2]
        
        print(f"\n[{section_name}] section in {file1}:")
        print("-" * 40)
        for i, atom in enumerate(parsed1[:10]):
            if 'id' in atom:
                print(f"{i+1:3d}: {atom['id']} {atom['type']} {atom['resnr']} {atom['resname']} "
                      f"{atom['atomname']} {atom['cgnr']} {atom['charge']:.4f} {atom['mass']:.4f}")
            else:
                print(f"{i+1:3d}: {atom['raw']}")
        if len(parsed1) > 10:
            print(f"... and {len(parsed1) - 10} more lines")
        
        print(f"\n[{section_name}] section in {file2}:")
        print("-" * 40)  
        for i, atom in enumerate(parsed2[:10]):
            if 'id' in atom:
                print(f"{i+1:3d}: {atom['id']} {atom['type']} {atom['resnr']} {atom['resname']} "
                      f"{atom['atomname']} {atom['cgnr']} {atom['charge']:.4f} {atom['mass']:.4f}")
            else:
                print(f"{i+1:3d}: {atom['raw']}")
        if len(parsed2) > 10:
            print(f"... and {len(parsed2) - 10} more lines")
            
        # Compare atom differences
        if len(parsed1) == len(parsed2):
            differences = []
            for i, (atom1, atom2) in enumerate(zip(parsed1, parsed2)):
                # First check if raw lines are identical
                if atom1['raw'].strip() != atom2['raw'].strip():
                    if 'id' in atom1 and 'id' in atom2:
                        if atom1['id'] != atom2['id']:
                            differences.append(f"Line {i+1}: ID {atom1['id']} vs {atom2['id']}")
                        elif atom1['type'] != atom2['type']:
                            differences.append(f"Line {i+1}: Type {atom1['type']} vs {atom2['type']}")
                        elif atom1['atomname'] != atom2['atomname']:
                            differences.append(f"Line {i+1}: Name {atom1['atomname']} vs {atom2['atomname']}")
                        elif atom1['charge'] != atom2['charge']:
                            differences.append(f"Line {i+1}: Charge {atom1['charge']} vs {atom2['charge']}")
                        elif atom1['mass'] != atom2['mass']:
                            differences.append(f"Line {i+1}: Mass {atom1['mass']} vs {atom2['mass']}")
                        else:
                            differences.append(f"Line {i+1}: Other difference")
                    else:
                        differences.append(f"Line {i+1}: Raw content differs")
            
            if differences:
                print(f"\nFound {len(differences)} atom differences:")
                for diff in differences[:10]:  # Show first 10
                    print(f"  {diff}")
                if len(differences) > 10:
                    print(f"  ... and {len(differences) - 10} more differences")
                    
                # Show actual line differences for first few
                print(f"\nFirst few line-by-line differences:")
                diff_count = 0
                for i, (atom1, atom2) in enumerate(zip(parsed1, parsed2)):
                    if atom1['raw'].strip() != atom2['raw'].strip() and diff_count < 3:
                        print(f"  Line {i+1}:")
                        print(f"    {file1}: {atom1['raw']}")
                        print(f"    {file2}: {atom2['raw']}")
                        diff_count += 1
            else:
                print("\n✓ All atoms match exactly!")
        else:
            print(f"\n✗ Different number of atoms: {len(parsed1)} vs {len(parsed2)}")
            if len(parsed1) > 0 and len(parsed2) > 0:
                # Show some entries from each to help identify the issue
                min_len = min(len(parsed1), len(parsed2))
                print(f"Comparing first {min_len} entries for differences:")
                for i in range(min(5, min_len)):
                    if parsed1[i]['raw'].strip() != parsed2[i]['raw'].strip():
                        print(f"  Line {i+1}:")
                        print(f"    {file1}: {parsed1[i]['raw']}")
                        print(f"    {file2}: {parsed2[i]['raw']}")
    
    else:
        # Handle bonded interactions (bonds, angles, dihedrals, etc.)
        parsed1 = [parse_interaction_line(line) for line in data1]
        parsed2 = [parse_interaction_line(line) for line in data2]
        
        print(f"\n[{section_name}] section in {file1}:")
        print("-" * 40)
        for i, interaction in enumerate(parsed1[:10]):
            atoms_str = ' '.join(map(str, interaction['atoms'])) if interaction['atoms'] else 'N/A'
            func_str = str(interaction['func_type']) if interaction['func_type'] is not None else 'N/A'
            params_str = ' '.join(interaction['params']) if interaction['params'] else ''
            print(f"{i+1:3d}: {atoms_str} {func_str} {params_str}")
        if len(parsed1) > 10:
            print(f"... and {len(parsed1) - 10} more lines")
        
        print(f"\n[{section_name}] section in {file2}:")
        print("-" * 40)
        for i, interaction in enumerate(parsed2[:10]):
            atoms_str = ' '.join(map(str, interaction['atoms'])) if interaction['atoms'] else 'N/A'
            func_str = str(interaction['func_type']) if interaction['func_type'] is not None else 'N/A'
            params_str = ' '.join(interaction['params']) if interaction['params'] else ''
            print(f"{i+1:3d}: {atoms_str} {func_str} {params_str}")
        if len(parsed2) > 10:
            print(f"... and {len(parsed2) - 10} more lines")
        
        # Compare interaction differences
        if len(parsed1) == len(parsed2):
            differences = []
            for i, (int1, int2) in enumerate(zip(parsed1, parsed2)):
                if int1['raw'].strip() != int2['raw'].strip():
                    if int1['atoms'] != int2['atoms']:
                        differences.append(f"Line {i+1}: Atoms {int1['atoms']} vs {int2['atoms']}")
                    elif int1['func_type'] != int2['func_type']:
                        differences.append(f"Line {i+1}: Function {int1['func_type']} vs {int2['func_type']}")
                    elif int1['params'] != int2['params']:
                        differences.append(f"Line {i+1}: Params {int1['params']} vs {int2['params']}")
                    else:
                        differences.append(f"Line {i+1}: Raw content differs")
            
            if differences:
                print(f"\nFound {len(differences)} {section_name} differences:")
                for diff in differences[:10]:  # Show first 10
                    print(f"  {diff}")
                if len(differences) > 10:
                    print(f"  ... and {len(differences) - 10} more differences")
                    
                # Show actual line differences for first few
                print(f"\nFirst few line-by-line differences:")
                diff_count = 0
                for i, (int1, int2) in enumerate(zip(parsed1, parsed2)):
                    if int1['raw'].strip() != int2['raw'].strip() and diff_count < 3:
                        print(f"  Line {i+1}:")
                        print(f"    {file1}: {int1['raw']}")
                        print(f"    {file2}: {int2['raw']}")
                        diff_count += 1
                        
            else:
                print(f"\n✓ All {section_name} match exactly!")
        else:
            print(f"\n✗ Different number of {section_name}: {len(parsed1)} vs {len(parsed2)}")
            if len(parsed1) > 0 and len(parsed2) > 0:
                # Show some entries from each to help identify the issue
                min_len = min(len(parsed1), len(parsed2))
                print(f"Comparing first {min_len} entries for differences:")
                for i in range(min(5, min_len)):
                    if parsed1[i]['raw'].strip() != parsed2[i]['raw'].strip():
                        print(f"  Line {i+1}:")
                        print(f"    {file1}: {parsed1[i]['raw']}")
                        print(f"    {file2}: {parsed2[i]['raw']}")

def show_section_details(filename, section_name):
    """Show details of a specific section."""
    sections = parse_itp_sections(filename)
    
    if section_name in sections:
        print(f"\n[{section_name}] section in {filename}:")
        print("-" * 40)
        for i, line in enumerate(sections[section_name][:10]):  # Show first 10 lines
            print(f"{i+1:3d}: {line}")
        if len(sections[section_name]) > 10:
            print(f"... and {len(sections[section_name]) - 10} more lines")
    else:
        print(f"Section [{section_name}] not found in {filename}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        file1 = sys.argv[1]  
        file2 = sys.argv[2]
    else:
        file1 = "../test_original.itp"
        file2 = "../test_output_standalone.itp"

    # Basic comparison
    compare_itp_files(file1, file2)
    
    # Get sections for detailed comparison
    sections1 = parse_itp_sections(file1)
    sections2 = parse_itp_sections(file2)
    
    # Define all possible topology sections to check
    all_topology_sections = [
        "defaults", "moleculetype", "atoms", "bonds", "pairs", "angles", 
        "dihedrals", "constraints", "settles", "exclusions", "position_restraints",
        "virtual_sites2", "virtual_sites3", "virtual_sites4", "virtual_sitesn",
        "system", "molecules", "atomtypes", "bondtypes", "angletypes", 
        "dihedraltypes", "constrainttypes", "nonbond_params", "pairtypes"
    ]
    
    # Find all sections present in either file
    present_sections = [section for section in all_topology_sections 
                       if section in sections1 or section in sections2]
    
    # Also include any non-standard sections
    all_found_sections = set(sections1.keys()) | set(sections2.keys())
    extra_sections = [section for section in all_found_sections 
                     if section not in all_topology_sections]
    
    all_sections_to_check = present_sections + sorted(extra_sections)
    
    print(f"\n=== Detailed Analysis Log ===")
    print(f"Will analyze {len(all_sections_to_check)} sections: {all_sections_to_check}")
    print()
    
    sections_with_differences = []
    sections_identical = []
    
    for section in all_sections_to_check:
        if section in sections1 or section in sections2:
            data1 = sections1.get(section, [])
            data2 = sections2.get(section, [])
            count1 = len(data1)
            count2 = len(data2)
            description = get_section_description(section)
            
            # Check if counts are different
            count_different = count1 != count2
            
            # Check if content is different (line by line comparison)
            content_different = False
            if count1 == count2 and count1 > 0:
                for i, (line1, line2) in enumerate(zip(data1, data2)):
                    if line1.strip() != line2.strip():
                        content_different = True
                        break
            
            # Determine overall status
            if count_different or content_different:
                sections_with_differences.append(section)
                status_msg = []
                if count_different:
                    status_msg.append(f"count: {count1} vs {count2}")
                if content_different:
                    status_msg.append("content differs")
                
                print(f"  • [{section:20s}]: DIFFERENT ({', '.join(status_msg)}) - {description}")
            else:
                sections_identical.append(section)
                print(f"  • [{section:20s}]: IDENTICAL ({count1:3d} entries) - {description}")
    
    print(f"\nSummary: {len(sections_identical)} identical, {len(sections_with_differences)} different")
    
    if sections_with_differences:
        print(f"Sections with differences: {sections_with_differences}")
    if sections_identical:
        print(f"Identical sections: {sections_identical}")
    
    # Show detailed comparison for all sections with differences
    if sections_with_differences:
        print(f"\n=== Detailed Analysis of Differing Sections ===")
        for section in sections_with_differences:
            compare_section_detailed(sections1, sections2, section, file1, file2)
    else:
        print(f"\n=== All Sections Identical! ===")
        print("All topology sections have identical entry counts.")
        # Still show a sample section for reference
        if "atoms" in sections1 and "atoms" in sections2:
            print("\nShowing atoms section as reference:")
            compare_section_detailed(sections1, sections2, "atoms", file1, file2)
