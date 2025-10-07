
from pathlib import Path
import pytest
import shutil
import sys
from reforge.mdsystem.gmxmd import GmxSystem, GmxRun

# Add the project root to Python path to find workflows
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from workflows import egfr_pipe

# Create a gmxSystem instance for testing.
sysdir = 'tests'
sysname = 'egfr_sys'
runname = 'egfr_run'
mdsys = GmxSystem(sysdir, sysname)
mdrun = GmxRun(sysdir, sysname, runname)
in_pdb = '../egfr_v3.pdb'

@pytest.fixture(scope="module", autouse=True)
def cleanup_test_files():
    """Setup and cleanup test files"""
    if mdsys.root.exists():
        shutil.rmtree(mdsys.root)
    yield mdsys  # This runs the tests
    shutil.rmtree(mdsys.root) # Cleanup - only runs if tests complete successfully

def test_setup():
    egfr_pipe.setup(sysdir, sysname)
    top_path = mdsys.root / "system.top"
    pdb_path = mdsys.root / "system.pdb"
    assert top_path.exists()
    assert pdb_path.exists()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])