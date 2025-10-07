#!/usr/bin/env python
"""
Hello World!
=================

Interface Tutorial

This example demonstrates how to use reForge's CLI interface for job submission and execution,
following the patterns used in the workflows directory.

Requirements:
    - HPC cluster with SLURM
    - Python 3.x
    - reForge package

Author: DY
"""

from pathlib import Path
import os

#%%
# One of the motivations behind reForge was to provide a user- and beginner-friendly interface
# for managing potentially hundreds or thousands of MD simulations without having to switch 
# between multiple scripts and constantly rewriting them—all while staying within the comfort of Python.
# This is what the 'cli' module is for:
from reforge.cli import run, sbatch, dojob, create_job_script

#%%
# Setup: Define your systems and runs (following the workflow pattern)
# The idea is simple: you have "n" systems and need to perform "m" independent runs 
# for each system to achieve sufficient sampling.
systems = [f'mutant_{i}' for i in range(3)]
runs = [f'run_{i}' for i in range(2)]

# Control whether to submit jobs to SLURM or run locally
submit = False  # Set to True to actually submit to SLURM

#%%
# Method 1: Direct execution with 'run'
# The run function executes commands directly in the current environment
print("=== Method 1: Direct execution ===")
for system in systems:
    for run_name in runs:
        # Execute the command (output goes to terminal)
        run(f'echo "Hello, I\'m {run_name} of {system}"')
        # Also print to show what the output would be
        print(f"Hello, I'm {run_name} of {system}")

#%%
# Method 2: Using the dojob pattern (like submit.py)
# This is the recommended approach for workflows that can run locally or on SLURM
print("=== Method 2: Using dojob pattern ===")

def dojob_example(submit, *args, **kwargs):
    """
    Submit a job if 'submit' is True; otherwise, run it via bash.
    This mirrors the pattern used in workflows/submit.py
    """
    # Set default SLURM parameters
    kwargs.setdefault('p', 'htc')        # partition
    kwargs.setdefault('q', 'public')     # qos
    kwargs.setdefault('t', '00:00:05')   # time limit
    kwargs.setdefault('mem', '50M')      # memory
    
    if submit:
        sbatch(*args, **kwargs)
    else:
        run('bash', *args)

# Execute jobs using the dojob pattern
for system in systems:
    for run_name in runs:
        dojob_example(submit, 'hello.sh', run_name, system)
        # Show what the hello.sh script would output
        print(f"Hello I am run {run_name} of system {system}")

#%%
# Method 3: Advanced workflow pattern with real workflow functions
# This demonstrates how to use actual workflow functions with the job submission system.
# We'll import mock functions from workflow.py to simulate a real MD workflow.
print("=== Method 3: Advanced workflow pattern with mock MD functions ===")

# Import mock workflow functions that simulate real MD tasks
from workflow import setup_system, run_md_simulation, analyze_trajectory, process_results

def sys_job_example(function_name, submit=False, **kwargs):
    """
    Submit or run a job for each system (like sys_job in submit.py).
    
    This function demonstrates the typical pattern where you have one task
    that needs to be executed once per system, such as:
    - System setup (creating input files)
    - Result processing (aggregating data from all runs)
    """
    print(f"\n--- Executing {function_name} for each system ---")
    for system in systems:
        if submit:
            # For submitted jobs, create a frozen script to avoid code changes during execution
            job_script = create_job_script('workflow.py', function_name, 'systems', system)
            dojob_example(submit, 'hello.sh', job_script, J=f'{function_name}_{system}', **kwargs)
            print(f"📝 Created job script and submitted: {function_name} for {system}")
        else:
            # For local runs, execute the workflow function directly
            if function_name == 'setup_system':
                setup_system('systems', system)
            elif function_name == 'process_results':
                process_results('systems', system)

def run_job_example(function_name, submit=False, **kwargs):
    """
    Submit or run a job for each system and run (like run_job in submit.py).
    
    This function demonstrates the pattern where you have tasks that need
    to be executed once per run of each system, such as:
    - MD simulations (one per run)
    - Trajectory analysis (one per run)
    """
    print(f"\n--- Executing {function_name} for each system and run ---")
    for system in systems:
        for run_name in runs:
            if submit:
                job_script = create_job_script('workflow.py', function_name, 'systems', system, run_name)
                dojob_example(submit, 'hello.sh', job_script, J=f'{function_name}_{system}_{run_name}', **kwargs)
                print(f"📝 Created job script and submitted: {function_name} for {run_name} of {system}")
            else:
                # For local runs, execute the workflow function directly
                if function_name == 'run_md_simulation':
                    run_md_simulation('systems', system, run_name)
                elif function_name == 'analyze_trajectory':
                    analyze_trajectory('systems', system, run_name)

# Example of a typical MD workflow execution order:
print("\n🔬 Simulating a typical MD workflow execution:")
print("1. First, set up all systems")
sys_job_example('setup_system', submit=submit)

print("\n2. Then, run MD simulations for all system-run combinations")
run_job_example('run_md_simulation', submit=submit)

print("\n3. Analyze trajectories for all runs")
run_job_example('analyze_trajectory', submit=submit)

print("\n4. Finally, process and aggregate results for each system")
sys_job_example('process_results', submit=submit)

#%%
# Now let's demonstrate what happens when we actually submit jobs (submit=True)
# This shows how frozen scripts are generated for SLURM submission
print("\n" + "="*60)
print("🚀 DEMONSTRATION: Job script creation with submit=True")
print("="*60)
print("""
When submit=True, reForge does the following:
1. Creates a 'frozen' copy of the workflow script
2. Generates a unique job directory with timestamp
3. Submits the job to SLURM with the frozen script
4. The frozen script ensures your code doesn't change during execution

Let's see this in action:
""")

# Enable submission to show script creation
submit_demo = True

# Demonstrate with a subset of the workflow
print("📋 Submitting setup jobs for all systems...")
sys_job_example('setup_system', submit=submit_demo, mem='500M', t='00:02:00')

print("\n📋 Submitting analysis jobs for first system only (to limit job count)...")
demo_systems_subset = [systems[0]]  # Just first system
demo_runs_subset = [runs[0]]  # Just first run

# Temporarily modify the systems for demo
original_systems = systems.copy()
systems[:] = demo_systems_subset
original_runs = runs.copy() 
runs[:] = demo_runs_subset

run_job_example('analyze_trajectory', submit=submit_demo, mem='1G', t='00:03:00')

# Restore original lists
systems[:] = original_systems
runs[:] = original_runs

# Show what was generated
print(f"\n📁 Checking generated job scripts...")
script_dir = Path('slurm_jobs')
if script_dir.exists():
    # Look for job directories created in the last few minutes
    job_dirs = [d for d in script_dir.iterdir() if d.is_dir()]
    recent_jobs = sorted(job_dirs, key=lambda x: x.stat().st_mtime, reverse=True)[:4]
    
    if recent_jobs:
        print(f"✅ Generated {len(recent_jobs)} recent job script directories:")
        for job_dir in recent_jobs:
            print(f"   📂 {job_dir.name}")
            script_files = list(job_dir.glob('*.py'))
            if script_files:
                print(f"      └── Contains frozen script: {script_files[0].name}")
    else:
        print("ℹ️  No recent job directories found")
else:
    print("ℹ️  slurm_jobs directory not found")

print(f"\n💡 These frozen scripts can now be executed by SLURM independently!")
print(f"   Each job runs with a snapshot of your code at submission time.")

#%%
# Summary: Key takeaways
print("=== Summary ===")
print("Key functions for job management:")
print("- run(): Execute commands directly")
print("- sbatch(): Submit jobs to SLURM")
print("- dojob(): Conditional execution (local vs SLURM)")
print("- create_job_script(): Create frozen scripts for submitted jobs")
print("\nWorkflow pattern:")
print("1. Define systems and runs")
print("2. Set submit=True/False to control execution mode")
print("3. Use dojob pattern for flexible execution")
print("4. Use sys_job/run_job patterns for complex workflows")