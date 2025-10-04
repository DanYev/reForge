# reForge

Documentation and instructions are available here: [reForge Documentation](https://danyev.github.io/reForge/)

## Martinize RNA Script

A self-contained script for converting all-atom RNA structures to coarse-grained Martini 3 representation.

### Usage
1. Download the script and additional `.itp` files from `scripts`.
2. Run:
```bash
python martinize_rna_v3.0.0.py --help
```

### Force Field Files

The script loads bonded parameters from:
`/path/to/the/script/martinize_rna_v3.0.0_itps`

The non-bonded parameters need to be included in the GROMACS topology file:
`scripts/martinize_rna_v3.0.0_itps/martini_v3.0.0_rna.itp`

### Citation

> Yangaliev D, Ozkan SB. Coarse-grained RNA model for the Martini 3 force field. Biophys J. 2025 Aug 5:S0006-3495(25)00483-7. doi: 10.1016/j.bpj.2025.07.034. 

### Copyright

Copyright (c) 2025, DY
