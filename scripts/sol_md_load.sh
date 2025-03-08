#!/bin/bash                     
# IMPORTANT! FOR SOL CLUSTER USERS! 
# RUN THE SCRIPT as ". sol_md_load.sh"
# with a dot (.) commands run in the current shell environment.
# NOT "bash sol_md_load.sh"
# "bash" starts a new shell process to execute the script.

module purge
module load mamba/latest
module load gromacs/2023.4-gpu-mpi
source activate reforge 

