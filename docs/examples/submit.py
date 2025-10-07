#!/usr/bin/env python
"""
Workflow Submission Example
===========================

This example demonstrates how to use reForge's submission system to manage
multiple MD simulations across different systems and runs. It shows both
local execution and SLURM job submission patterns.

The workflow uses mock functions from workflow.py to demonstrate typical
MD pipeline operations without requiring actual MD software.
"""

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
    """Submit or run a job for each system.
    
    This function demonstrates the 'sys_job' pattern used in production workflows.
    It executes a function once per system, which is typical for setup operations
    or analyses that operate on the entire system.
    """
    print(f"\n=== Running sys_job: {function} (submit={submit}) ===")
    for sysname in sysnames:
        if submit:
            # Create a job-specific script to freeze the code at submission time
            job_script = create_job_script(pyscript, function, sysdir, sysname)
            print(f"📝 Created frozen script: {Path(job_script).name}")
            dojob(submit, shscript, job_script, J=f'{function}_{sysname}', **kwargs)
        else:
            print(f"🔄 Executing locally: {function} for {sysname}")
            dojob(submit, shscript, pyscript, function, sysdir, sysname, 
                  J=f'{function}_{sysname}', **kwargs)


def run_job(function, submit=False, **kwargs):
    """Submit or run a job for each system and run combination.
    
    This function demonstrates the 'run_job' pattern used in production workflows.
    It executes a function for every combination of system and run, which is typical
    for MD simulations or analyses that need to be performed on each trajectory.
    """
    print(f"\n=== Running run_job: {function} (submit={submit}) ===")
    for sysname in sysnames:
        for runname in runs:
            if submit:
                # Create a job-specific script to freeze the code at submission time
                job_script = create_job_script(pyscript, function, sysdir, sysname, runname)
                print(f"📝 Created frozen script: {Path(job_script).name}")
                dojob(submit, shscript, job_script, J=f'{function}_{sysname}_{runname}', **kwargs)
            else:
                print(f"🔄 Executing locally: {function} for {sysname}/{runname}")
                dojob(submit, shscript, pyscript, function, sysdir, sysname, runname,
                      J=f'{function}_{sysname}_{runname}', **kwargs)


#%%
# Configuration and Setup
# This section defines the workflow parameters and demonstrates how to structure
# a typical reForge workflow submission script.

if __name__ == "__main__":
    # Setup paths and scripts
    # Use current directory for Sphinx-Gallery compatibility
    pdir = Path('.')
    shscript = str(pdir / 'run.sh')  # Shell script that runs Python functions
    pyscript = str(pdir / 'workflow.py')  # Python script with workflow functions

    # Define systems and runs for the workflow
    sysdir = 'systems'  # Directory containing system files
    sysnames = ['mutant_1', 'mutant_2']  # Systems to process
    runs = ["mdrun_1", "mdrun_2"]  # Simulation runs for each system

    # Control submission mode
    submit = False  # Set to True to submit to SLURM, False for local execution

    print("🚀 Starting reForge Workflow Demonstration")
    print(f"📁 Working directory: {pdir}")
    print(f"🔧 Shell script: run.sh")
    print(f"🐍 Python workflow: workflow.py")
    print(f"💾 Systems: {sysnames}")
    print(f"🏃 Runs: {runs}")
    print(f"📤 Submit mode: {submit}")

    #%%
    # Molecular Dynamics Workflow
    # This section demonstrates typical MD workflow steps
    
    print("\n" + "="*50)
    print("🧬 MOLECULAR DYNAMICS WORKFLOW")
    print("="*50)
    
    # System setup (one job per system)
    sys_job('setup_system', submit=submit)
    
    # Run MD simulations (one job per system-run combination)
    run_job('run_md_simulation', submit=submit, mem='4G', t='00-01:00:00')
    
    #%%
    # Analysis Workflow
    # This section demonstrates typical analysis steps
    
    print("\n" + "="*50)
    print("📊 ANALYSIS WORKFLOW")
    print("="*50)
    
    # Trajectory analysis (one job per system-run combination)
    run_job('analyze_trajectory', submit=submit, mem='2G', t='00-00:30:00')
    
    # Results processing (one job per system)
    sys_job('process_results', submit=submit, mem='1G', t='00-00:15:00')
    
    print("\n" + "="*50)
    print("✅ Workflow demonstration complete!")
    print("="*50)
