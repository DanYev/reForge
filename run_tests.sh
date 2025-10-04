#!/bin/bash
#SBATCH --time=0-00:10:00                                                       
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH -o tests/sl_output.out
#SBATCH -e tests/sl_error.err
# -----------------------------------------------------------------------------
# run_tests.sh
#
# Description:
#   This script runs all unit tests for the project using pytest.
#   Some tests need GPU and some - installed GROMACS.
#   Ideally, you want to run them in the interactive mode,
#   but the last time I tried gromacs did not work in it both on PHX and SOL.
#
# Usage:
#   From the project root, run:
#       ./run_tests.sh
#   or
#       sbatch run_tests.sh
#
# Requirements:
#   - Python 3.x and pytest must be installed.
#   - CUDA
#   - GROMACS
# -----------------------------------------------------------------------------

# # Get the directory of this script and change to it (assumed to be the project root)
# SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# cd "$SCRIPT_DIR"

if [ "$1" == "--all" ]; then
    echo "Running all tests..."
    pytest --maxfail=1 --disable-warnings -q
else
    pytest -v tests/test_rpymath.py --maxfail=1 --disable-warnings -q
    pytest -v tests/test_rcmath.py --maxfail=1 --disable-warnings -q
    pytest -v tests/test_mdm.py --maxfail=1 --disable-warnings -q
    pytest -v tests/test_pdbtools.py --maxfail=1 --disable-warnings -q
    pytest -v tests/test_gmxmd.py --maxfail=1 --disable-warnings -q
    pytest -v tests/test_martinize.py --maxfail=1 --disable-warnings -q
fi

# ghp-import -n -p -f build/html