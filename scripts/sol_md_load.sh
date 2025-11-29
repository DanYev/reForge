#!/bin/bash                     
# IMPORTANT! FOR SOL CLUSTER USERS! 
# RUN THE SCRIPT as ". sol_md_load.sh"
# with a dot (.) commands run in the current shell environment.
# NOT "bash sol_md_load.sh"
# "bash" starts a new shell process to execute the script.

module purge
module load mamba/latest
module load gromacs
source activate reforge

# Pre-load reforge to speed up interactive usage (optional)
# Comment out if causing issues
# echo "Pre-loading reforge modules..."
# python -c "import reforge; print('✓ reforge pre-loaded and ready')" 2>/dev/null || echo "⚠ reforge import failed" 

