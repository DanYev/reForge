from reforge.cli import sbatch, run


def dojob(submit, *args, **kwargs):
    """
    Submit a job if 'submit' is True; otherwise, run it via bash.
    
    Parameters:
        submit (bool): Whether to submit (True) or run (False) the job.
        *args: Positional arguments for the job.
        **kwargs: Keyword arguments for the job.
    """
    if submit:
        sbatch(*args, **kwargs)
    else:
        run('bash', *args)


def setup(submit=False, md_module='mm_md', **kwargs): 
    """
    Set up the md model for each system name.
    
    Parameters:
        submit (bool): Whether to submit the job.
        **kwargs: Additional keyword arguments for the job.
    """
    kwargs.setdefault('mem', '3G')
    for sysname in sysnames:
        dojob(submit, shscript, pyscript, md_module, 'setup', sysdir, sysname, 
              J=f'setup_{sysname}', **kwargs)


def md(submit=True, md_module='mm_md', ntomp=8, **kwargs):
    """
    Run molecular dynamics simulations for each system and mdrun.
    
    Parameters:
        submit (bool): Whether to submit the job.
        ntomp (int): Number of OpenMP threads.
        **kwargs: Additional keyword arguments.
    """
    kwargs.setdefault('t', '00-04:00:00')
    kwargs.setdefault('N', '1')
    kwargs.setdefault('n', '1')
    kwargs.setdefault('c', ntomp)
    # kwargs.setdefault('G', '1')
    kwargs.setdefault('mem', '3G')
    kwargs.setdefault('e', 'slurm_output/error.%A.err')
    kwargs.setdefault('o', 'slurm_output/output.%A.out')
    for sysname in sysnames:
        for runname in runs:
            dojob(submit, shscript, pyscript, md_module, 'md', sysdir, sysname, runname, ntomp, 
                  J=f'md_{sysname}', **kwargs)


def extend(submit=True, md_module='mm_md', ntomp=8, **kwargs):
    """
    Extend simulations by processing each system and mdrun.
    
    Parameters:
        submit (bool): Whether to submit the job.
        ntomp (int): Number of OpenMP threads.
        **kwargs: Additional keyword arguments.
    """
    kwargs.setdefault('t', '00-04:00:00')
    kwargs.setdefault('N', '1')
    kwargs.setdefault('n', '1')
    kwargs.setdefault('c', ntomp)
    kwargs.setdefault('G', '1')
    kwargs.setdefault('mem', '3G')
    kwargs.setdefault('e', 'slurm_output/error.%A.err')
    kwargs.setdefault('o', 'slurm_output/output.%A.out')
    for sysname in sysnames:
        for runname in runs:
            dojob(submit, shscript, pyscript, md_module, 'extend', sysdir, sysname, runname, ntomp, 
                  J=f'ext_{sysname}', **kwargs)
                

def trjconv(submit=True, md_module='mm_md', **kwargs):
    """
    Convert trajectories for each system and mdrun.
    
    Parameters:
        submit (bool): Whether to submit the job.
        **kwargs: Additional keyword arguments.
    """
    kwargs.setdefault('t', '00-01:00:00')
    kwargs.setdefault('mem', '2G')
    for sysname in sysnames:
        for runname in runs:
            dojob(submit, shscript, pyscript, md_module, 'trjconv', sysdir, sysname, runname,
                  J=f'trjconv_{sysname}', **kwargs)

            
def rms_analysis(submit=True, **kwargs):
    """
    Perform RMSD analysis for each system and mdrun.
    
    Parameters:
        submit (bool): Whether to submit the job.
        **kwargs: Additional keyword arguments.
    """
    for sysname in sysnames:
        for runname in runs:
            dojob(submit, shscript, pyscript, 'analysis', 'rms_analysis', sysdir, sysname, runname,
                  J=f'rms_{sysname}', **kwargs)
      

def cov_analysis(submit=True, **kwargs):
    """
    Perform covariance analysis for each system and mdrun.
    
    Parameters:
        submit (bool): Whether to submit the job.
        **kwargs: Additional keyword arguments.
    """
    kwargs.setdefault('t', '00-04:00:00')
    kwargs.setdefault('mem', '25G')
    for sysname in sysnames:
        for runname in runs:
            dojob(submit, shscript, pyscript, 'analysis', 'cov_analysis', sysdir, sysname, runname,
                  J=f'cov_{sysname}', **kwargs)


def cluster(submit=True, **kwargs):
    """
    Run clustering analysis for each system and mdrun.
    
    Parameters:
        submit (bool): Whether to submit the job.
        **kwargs: Additional keyword arguments.
    """
    for sysname in sysnames:
        for runname in runs:
            dojob(submit, shscript, pyscript, 'analysis', 'cluster', sysdir, sysname, runname,
                  J=f'cluster_{sysname}', **kwargs)                    


def tdlrt_analysis(submit=True, **kwargs):
    """Perform tdlrt analysis for each system and run."""
    kwargs.setdefault('t', '00-01:00:00')
    kwargs.setdefault('mem', '30G')
    kwargs.setdefault('G', '1')
    for sysname in sysnames:
        for runname in runs:
            dojob(submit, shscript, pyscript, 'analysis', 'tdlrt_analysis', sysdir, sysname, runname,
                  J=f'tdlrt_{sysname}', **kwargs)


def tdlrt_figs(submit=True, **kwargs):
    """Generate figures from tdlrt analysis for each system and run."""
    kwargs.setdefault('mem', '7G')
    for sysname in sysnames:
        for runname in runs:
            dojob(submit, shscript, pyscript, 'analysis', 'tdlrt_figs', sysdir, sysname, runname,
                  J=f'plot_{sysname}', **kwargs)
 

def get_averages(submit=False, **kwargs):
    """Calculate average values for each system."""
    kwargs.setdefault('mem', '80G')
    for sysname in sysnames:
        dojob(submit, shscript, pyscript, 'analysis', 'get_averages', sysdir, sysname, 
              J=f'av_{sysname}', **kwargs)


def get_td_averages(submit=False, **kwargs):
    """Calculate time-dependent averages for each system."""
    kwargs.setdefault('mem', '80G')
    for sysname in sysnames:
        dojob(submit, shscript, pyscript, 'analysis', 'get_td_averages', sysdir, sysname, 
              J=f'tdav_{sysname}', **kwargs)   


def plot(submit=False, **kwargs):
    """Generate plots for each system."""
    for sysname in sysnames:
        dojob(submit, shscript, 'plot.py', module, sysdir, sysname, 
              J='plotting', **kwargs)


def sys_job(module, jobname, submit=False, **kwargs):
    """
    Submit or run a system-level job for each system.
    
    Parameters:
        jobname (str): The name of the job.
        submit (bool): Whether to submit the job.
        **kwargs: Additional keyword arguments.
    """
    for sysname in sysnames:
        dojob(submit, shscript, pyscript, module, jobname, sysdir, sysname, 
              J=f'{jobname}', **kwargs)


def run_job(module, jobname, submit=False, **kwargs):
    """
    Submit or run a run-level job for each system and run.
    
    Parameters:
        jobname (str): The name of the job.
        submit (bool): Whether to submit the job.
        **kwargs: Additional keyword arguments.
    """
    kwargs.setdefault('t', '00-02:00:00')
    kwargs.setdefault('mem', '17G')
    for sysname in sysnames:
        for runname in runs:
            dojob(submit, shscript, pyscript, module, jobname, sysdir, sysname, runname,
                  J=f'{jobname}', **kwargs)


MARTINI = False

if __name__ == "__main__":
    shscript = 'sbatch.sh'
    pyscript = 'exec.py'
    md_module = 'mm_md'

    sysdir = 'systems_kras' 
    sysnames = ['emu'] 
    runs = ['mdrun_1', 'mdrun_2', ]

    # sysdir = 'systems' 
    # sysnames = ['1btl', '1btl_go_2', '1btl_go_3'] 
    # runs = ['mdrun_1', 'mdrun_2', 'mdrun_3', 'mdrun_4']

    # setup(submit=False, md_module=md_module, mem='4G')
    # md(submit=True, md_module=md_module, ntomp=4, mem='2G', q='public', p='htc', t='00-01:00:00', G=1)
    # md(submit=True, md_module=md_module, ntomp=8, mem='2G', q='grp_sozkan', p='general', t='01-00:00:00', G=1)
    # extend(submit=True, md_module=md_module, ntomp=8, mem='2G', q='public', p='htc', t='00-04:00:00', G=1)
    # extend(submit=True, md_module=md_module, ntomp=8, mem='2G', q='grp_sozkan', p='general', t='01-00:00:00', G=1)
    run_job('mm_md', 'sample_emu', submit=True, q='public', p='htc', t='00-04:00:00', G=1)
    # trjconv(submit=True, md_module=md_module, t='00-01:00:00', q='public', p='htc', c='1', mem='2G')
    # rms_analysis(submit=True, mem='4G')
    # get_td_averages(submit=True, mem='32G')
    # plot(submit=False)
    # cluster(submit=True)
    # tdlrt_analysis(submit=False)
    # tdlrt_figs(submit=True)
    # test(submit=True)
    # sys_job('analysis', 'pca_trajs', submit=False)
    # sys_job('analysis', 'clust_cov', submit=False)
    # cov_analysis(submit=False, mem='4G')
    # get_averages(submit=False, mem='4G') 
    # run_job('do_run_pca', submit=True, q='public', p='htc', t='00-01:00:00')
