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

The script loads some of the bonded parameters from:
`/path/to/the/script/martinize_rna_v3.0.0_itps`

The non-bonded parameters need to be included in the GROMACS topology file:
`scripts/martinize_rna_v3.0.0_itps/martini_v3.0.0_rna.itp`

```

### Copyright

Copyright (c) 2025, DY
