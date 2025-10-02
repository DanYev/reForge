# reForge

Documentation and instructions are available here: [reForge Documentation](https://danyev.github.io/reForge/)

## Martinize RNA Standalone Script

A self-contained script for converting all-atom RNA structures to coarse-grained Martini 3.0 representation.

### Usage

```bash
# Usage
python martinize_rna_standalone --help

```

### Requirements

- Python 3.7+
- NumPy

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-f` | Required | Input PDB file |
| `-ot` | `molecule.itp` | Output topology file |
| `-os` | `molecule.pdb` | Output CG structure |
| `-mol` | `molecule` | Molecule name |
| `-elastic` | `yes` | Add elastic network |
| `-ef` | `200` | Elastic force constant |
| `--debug` | `False` | Enable debug logging |

### Force Field Files

The script automatically loads Martini 3.0 RNA force field parameters from:
`reforge/forge/forcefields/rna_reg/`

- **rna_A_new.itp** - Adenine nucleotide parameters
- **rna_C_new.itp** - Cytosine nucleotide parameters  
- **rna_G_new.itp** - Guanine nucleotide parameters
- **rna_U_new.itp** - Uracil nucleotide parameters

```

### Copyright

Copyright (c) 2025, DY
