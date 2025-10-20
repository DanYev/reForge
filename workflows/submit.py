import os
from pathlib import Path
import shutil
from reforge.cli import sbatch, run, run_command, create_job_script


def dojob(submit, *args, **kwargs):
    """
    Submit a job if 'submit' is True; otherwise, run it via bash.
    
    Parameters:
        submit (bool): Whether to submit (True) or run (False) the job.
        *args: Positional arguments for the job.
        **kwargs: Keyword arguments for the job.
    """
    kwargs.setdefault('p', 'htc')
    kwargs.setdefault('q', 'public')
    kwargs.setdefault('t', '00-00:20:00')
    kwargs.setdefault('N', '1')
    kwargs.setdefault('n', '1')
    kwargs.setdefault('mem', '2G')
    if submit:
        sbatch(*args, **kwargs)
    else:
        run('bash', *args)

            
def sys_job(function, submit=False, **kwargs):
    """Submit or run a job for each system."""
    for sysname in sysnames:
        if submit:
            # Create a job-specific script to freeze the code at submission time
            job_script = create_job_script(pyscript, function, sysdir, sysname)
            dojob(submit, shscript, job_script, J=f'{function}', **kwargs)
        else:
            dojob(submit, shscript, pyscript, function, sysdir, sysname, 
                  J=f'{function}', **kwargs)


def run_job(function, submit=False, **kwargs):
    """Submit or run a job for each system and run."""
    for sysname in sysnames:
        for runname in runs:
            if submit:
                # Create a job-specific script to freeze the code at submission time
                job_script = create_job_script(pyscript, function, sysdir, sysname, runname)
                dojob(submit, shscript, job_script, J=f'{function}', **kwargs)
            else:
                dojob(submit, shscript, pyscript, function, sysdir, sysname, runname,
                      J=f'{function}', **kwargs)


def prepare_systems():
    for item in INPUT_DIR.iterdir():
        if item.is_file() and item.suffix == ".pdb":
            dest = MDSYS_DIR / item.stem
            dest.mkdir(exist_ok=True, parents=True)
            shutil.copy(str(item), str(dest / 'input.pdb'))


if __name__ == "__main__":
    INPUT_DIR = Path("../tests/")
    MDSYS_DIR = Path("systems")
    prepare_systems()

    pdir = Path(__file__).parent
    shscript = str(pdir / 'run.sh')

    sysdir = str(MDSYS_DIR)
    sysnames = os.listdir(sysdir)
    runs = ["mdrun_1"]

    submit = True

    ##### For MD #####
    pyscript = str(pdir / 'gmx_md.py')
    run_job('workflow', submit=submit, G='1', c='8', mem='2G', t='00-02:00:00')
    # sys_job('setup', submit=submit)
    # # run_job('md_npt', submit=submit, G='1', c='4', mem='2G', t='00-02:00:00')
    # # run_job('extend', submit=submit, G='1', c='4', mem='2G')
    # # run_job('trjconv', submit=submit)
    # run_job('main', submit=submit)

    ##### Analysis #####
    # pyscript = str(pdir / 'common.py')
    # sys_job('pca_trajs', submit=submit) # PCA
    # sys_job('clust_cov', submit=submit) # Clustering
    # run_job('rms_analysis', submit=submit) # RMSF/RMSD
    # run_job('cov_analysis', submit=submit) # DFI/DCI
    # sys_job('get_means_sems', submit=submit) 
    # run_job('tdlrt_analysis', submit=submit) # TDLRT
    # sys_job('get_averages', submit=submit, c='1', mem='4G') # Big arrays: mem > 2 * c * array size
    # sys_job('enm_analysis', submit=submit, G='1', mem='8G') # ENM

