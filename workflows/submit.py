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


MARTINI = False

if __name__ == "__main__":
    pdir = Path(__file__).parent
    shscript = str(pdir / 'run.sh')
    pyscript = str(pdir / 'mm_md.py')
    print(f"Using script: {pyscript}")

    sysdir = 'tests/test' 
    sysnames = ['sys_test'] 
    runs = ['run_test']

    # sys_job('setup', submit=False)
    run_job('md_npt', submit=True, G='1')
