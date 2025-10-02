from pathlib import Path
import shutil
import tempfile
import datetime
import sys
import inspect
from reforge.cli import sbatch, run


def run_command():
    """
    Automatically discover and run functions from command line arguments.
    This eliminates the need to manually maintain a function mapping.
    Can be imported and used by any workflow script.
    """
    if len(sys.argv) < 2:
        module = sys.modules['__main__']  # Get the main module (the script being run)
        module_name = getattr(module, '__name__', sys.argv[0])
        # Get all public functions (not starting with _)
        functions = {name: obj for name, obj in inspect.getmembers(module, inspect.isfunction) 
                    if not name.startswith('_')}
        print(f"Usage: python {sys.argv[0]} <function_name> [args...]", file=sys.stderr)
        print(f"Available functions: {', '.join(sorted(functions.keys()))}", file=sys.stderr)
        sys.exit(1)
    
    command = sys.argv[1]
    args = sys.argv[2:]
    
    # Get main module and discover all public functions
    module = sys.modules['__main__']
    module_name = getattr(module, '__name__', sys.argv[0])
    functions = {name: obj for name, obj in inspect.getmembers(module, inspect.isfunction) 
                if not name.startswith('_')}
    
    if command not in functions:
        print(f"Error: Unknown function '{command}'", file=sys.stderr)
        print(f"Available functions for {module_name}: {', '.join(sorted(functions.keys()))}", file=sys.stderr)
        sys.exit(1)
    
    try:
        print(f"Calling {command} with args: {args}", file=sys.stderr)
        if args:
            functions[command](*args)
        else:
            functions[command]()
        print(f"Successfully completed {command}", file=sys.stderr)
    except Exception as e:
        print(f"Error executing {command}: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


def create_job_script(original_script, function, *args):
    """
    Create a standalone job script that captures the function call at submission time.
    This prevents issues when the original script is modified after job submission.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    job_dir = Path("slurm_jobs") / f"job_{function}_{timestamp}_{hash(str(args)) % 10000}"
    job_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy the original script to preserve the version at submission time
    script_copy = job_dir / f"script_{timestamp}.py"
    shutil.copy2(original_script, script_copy)
    
    # Create a wrapper script that calls the specific function
    wrapper_script = job_dir / f"wrapper_{timestamp}.py"
    
    with open(wrapper_script, 'w') as f:
        f.write(f'''#!/usr/bin/env python
"""
Auto-generated wrapper script for function: {function}
Created at: {datetime.datetime.now()}
Original script: {original_script}
Arguments: {args}
"""

import sys
import os
from pathlib import Path

# Add the script directory to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

# Import the copied script
import {script_copy.stem} as target_module

# Add the original script's directory to path to import run_command
original_script_path = Path("{original_script}")
original_script_dir = original_script_path.parent
sys.path.insert(0, str(original_script_dir))

if __name__ == "__main__":
    # Set up sys.argv to mimic command line call
    function_name = "{function}"
    args = {list(args)}
    sys.argv = [__file__, function_name] + args
    
    # Try to use the centralized run_command function
    try:
        from submit import run_command
        # Temporarily set the main module to our target module
        sys.modules['__main__'] = target_module
        run_command()
    except ImportError:
        # Fall back to target module's _run_command if available
        if hasattr(target_module, '_run_command'):
            target_module._run_command()
        else:
            # Final fallback to direct function calling
            if hasattr(target_module, function_name):
                func = getattr(target_module, function_name)
                print(f"Calling {{function_name}} with args: {{args}}", file=sys.stderr)
                try:
                    if args:
                        func(*args)
                    else:
                        func()
                    print(f"Successfully completed {{function_name}}", file=sys.stderr)
                except Exception as e:
                    print(f"Error executing {{function_name}}: {{str(e)}}", file=sys.stderr)
                    import traceback
                    traceback.print_exc(file=sys.stderr)
                    sys.exit(1)
            else:
                print(f"Function '{{function_name}}' not found in module", file=sys.stderr)
                sys.exit(1)
''')
    
    return str(wrapper_script)


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
            # For local runs, use the current script directly
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
                # For local runs, use the current script directly
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

    # Example usage:
    sys_job('setup', submit=True)
    
    # To submit jobs to the queue (preserving script version):
    # sys_job('setup', submit=True, t='00-01:00:00', mem='4G')
    
    # For single jobs:
    # single_job('setup', 'specific_system', submit=True)
    # single_job('md_npt', 'sys1', 'run1', submit=True, t='00-04:00:00', mem='8G')
    
