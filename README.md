# reForge

Documentation and instructions are available here: [reForge Documentation](https://danyev.github.io/reForge/)

**For Ozkan Lab Users:** The main branch is updated only after all example workflows pass comprehensive testing.

**Troubleshooting:**
- First, ensure your local repository and environment are up-to-date
- If issues persist, please provide error logs, the script you're running, and relevant input files when reporting problems

## Martinize RNA Script

A standalone script that converts all-atom RNA structures to coarse-grained representations using the Martini 3 force field.

### Usage
1. Download the script and required `.itp` files from the `scripts/` directory
2. View available options:
```bash
python martinize_rna_v3.0.0.py --help
```

### Force Field Files

**Bonded Parameters:** The script automatically loads bonded parameters from:
```
/path/to/the/script/martinize_rna_v3.0.0_itps/
```

**Non-bonded Parameters:** Include the following file in your GROMACS topology:
```
scripts/martinize_rna_v3.0.0_itps/martini_v3.0.0_rna.itp
```

### Citation

If you use this script in your research, please cite:

> Yangaliev D, Ozkan SB. Coarse-grained RNA model for the Martini 3 force field. *Biophys J*. 2025 Aug 5:S0006-3495(25)00483-7. doi: [10.1016/j.bpj.2025.07.034](https://doi.org/10.1016/j.bpj.2025.07.034)

### License

Copyright (c) 2025, Danis Yangaliev  
Licensed under the GNU General Public License v3.0
