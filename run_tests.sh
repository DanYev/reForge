#!/bin/bash
#SBATCH --time=0-00:15:00                                                       
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

# Initialize test result tracking
TEST_EXIT_CODE=0
FAILED_TESTS=""

if [ "$1" == "--all" ]; then
    echo "Running all tests..."
    pytest --maxfail=1 --disable-warnings -q
    TEST_EXIT_CODE=$?
else
    echo "Running individual tests..."
    
    # Array of test files to run
    TEST_FILES=(
        # "tests/test_rpymath.py"
        # "tests/test_rcmath.py" 
        # "tests/test_mdm.py"
        # "tests/test_pdbtools.py"
        # "tests/test_martinize.py"
        # "tests/test_gmxmd.py"
        "tests/test_mmmd.py"
        "tests/test_common.py"
    )
    
    # Run each test and track results
    for test_file in "${TEST_FILES[@]}"; do
        echo "Running $test_file..."
        pytest -v "$test_file" --maxfail=1 --disable-warnings -q
        EXIT_CODE=$?
        
        if [ $EXIT_CODE -ne 0 ]; then
            TEST_EXIT_CODE=$EXIT_CODE
            FAILED_TESTS="$FAILED_TESTS $test_file"
            echo "❌ FAILED: $test_file"
        else
            echo "✅ PASSED: $test_file"
        fi
    done
fi

# Report final results
echo ""
echo "=================================="
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "🎉 ALL TESTS PASSED! 🎉"
    echo "Test run completed successfully."
else
    echo "❌ SOME TESTS FAILED!"
    echo "Failed tests:$FAILED_TESTS"
    echo "Exit code: $TEST_EXIT_CODE"
fi
echo "=================================="

# Uncomment if you want to build docs only on successful tests
# if [ $TEST_EXIT_CODE -eq 0 ]; then
#     echo "Tests passed, building documentation..."
#     ghp-import -n -p -f build/html
# else
#     echo "Skipping documentation build due to test failures."
# fi

# Exit with the test result code so SLURM/CI can detect failures
exit $TEST_EXIT_CODE