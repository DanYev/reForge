from pathlib import Path
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


if __name__ == "__main__":
    pdir = Path(__file__).parent
    shscript = str(pdir / 'run.sh')

    sysdir = 'systems' 
    sysnames = ['enm_system'] 
    runs = ["nm_run"]

    submit = False

    ##### For MD #####
    pyscript = str(pdir / 'egfr_pipe.py')
    run_job('main', submit=submit)
    # sys_job('setup', submit=submit)
    # run_job('md_npt', submit=submit, G='1', c='4', mem='2G', t='00-02:00:00')
    # run_job('extend', submit=submit, G='1', c='4', mem='2G')
    # run_job('trjconv', submit=submit)

    ##### Analysis #####
    # pyscript = str(pdir / 'common.py')
    # sys_job('pca_trajs', submit=submit) # PCA
    # sys_job('clust_cov', submit=submit) # Clustering
    # run_job('rms_analysis', submit=submit) # RMSF/RMSD
    # run_job('cov_analysis', submit=submit) # DFI/DCI
    # sys_job('get_means_sems', submit=submit) 
    # run_job('tdlrt_analysis', submit=submit) # TDLRT
    # sys_job('get_averages', submit=submit, c='1', mem='4G') # Big arrays: mem > 2 * c * array size
