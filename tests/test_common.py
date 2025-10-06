"""
Test Suite for workflows.common Package
=====================================
"""

from pathlib import Path
import pytest
import shutil
import sys
import numpy as np
from reforge.mdsystem.mdsystem import MDSystem, MDRun

# Add the project root to Python path to find workflows
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from workflows import common

# Create test system instances
sysdir = 'tests'
sysname = 'mm_sys'
runname = 'mm_run'
mdsys = MDSystem(sysdir, sysname)
mdrun = MDRun(sysdir, sysname, runname)

@pytest.fixture(scope="module", autouse=True)
def cleanup_test_files():
    """Setup and cleanup test files"""
    yield mdsys  # This runs the tests
    if mdsys.root.exists():  # Cleanup - only runs if tests complete successfully
        shutil.rmtree(mdsys.root)

def test_pca_trajs():
    common.pca_trajs(sysdir, sysname)
    assert (mdsys.datdir / "cluster_0.xtc").exists()

def test_clust_cov():
    common.clust_cov(sysdir, sysname)
    assert (mdsys.datdir / "cdfi_filt_av.npy").exists()

def test_cov_analysis():
    common.cov_analysis(sysdir, sysname, runname)
    assert (mdrun.covdir / "dfi_1.npy").exists()

def test_rms_analysis():
    common.rms_analysis(sysdir, sysname, runname)
    assert (mdrun.rmsdir / 'rmsf_values.npy').exists()
    assert (mdrun.rmsdir / 'rmsd_values.npy').exists()

def test_get_means_sems():
    common.get_means_sems(sysdir, sysname)
    assert (mdsys.datdir / 'dfi_av.npy').exists()
    assert (mdsys.datdir / 'dfi_err.npy').exists()


# def test_tdlrt_analysis():
#     """Test time-dependent linear response theory analysis"""
#     common.tdlrt_analysis(sysdir, sysname, runname)


# def test_get_averages():
#     """Test calculation of averages from patterns"""
#     result = common.get_averages(sysdir, sysname, pattern="*.npy")
#     # Should handle case where no files match pattern


# def test_sample_emu():
#     """Test enhanced sampling from EMU"""
#     common.sample_emu(sysdir, sysname, runname)


# def test_pdb_to_seq():
#     """Test PDB to sequence conversion"""
#     # Create a simple test PDB file
#     test_pdb = Path("test_protein.pdb")
#     try:
#         with open(test_pdb, 'w') as f:
#             f.write("ATOM      1  CA  ALA A   1      10.000  10.000  10.000  1.00 20.00           C\n")
#             f.write("ATOM      2  CA  GLY A   2      11.000  10.000  10.000  1.00 20.00           C\n")
#             f.write("END\n")
        
#         sequence = common._pdb_to_seq(str(test_pdb))
#         assert isinstance(sequence, str)
#         # Should convert ALA GLY to AG
#         assert "AG" in sequence or len(sequence) >= 2
        
#     finally:
#         # Cleanup test file
#         if test_pdb.exists():
#             test_pdb.unlink()


# def test_selection_constants():
#     """Test that selection constants are properly defined"""
#     assert hasattr(common, 'SELECTION')
#     assert hasattr(common, 'TRJEXT')
#     assert isinstance(common.SELECTION, str)
#     assert isinstance(common.TRJEXT, str)
#     assert common.TRJEXT in ['trr', 'xtc']


# def test_process_batch():
#     """Test batch processing function"""
#     # Create some test data
#     test_data = np.random.rand(10, 5)
#     test_args = (test_data, "test_batch")
    
#     result = common._process_batch(test_args)
#     # Should return some processed result
#     assert result is not None


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])
