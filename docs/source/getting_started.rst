Getting Started
===============

Installation
------------

The "installation" of reForge consists of including the reForge project directory 
in your Python path and ensuring that you have all the necessary dependencies.
All the packages that reForge depends on, except for GROMACS, can be installed with conda and/or pip.

.. warning::

    For users of SOL and PHX clusters:
    For more detailed instructions, scroll down the page.

1. **Clone the repository:**

    This will create the directory "reforge" in your current directory.

.. code-block:: bash

    git clone https://github.com/DanYev/reforge.git

2. **Install the virtual environment:**

.. code-block:: bash

    cd reforge 
    conda env create -n reforge --file environment.yml
    source activate reforge

3. **Include reForge in your Python path OR install it via pip:**

.. code-block:: bash

    export PYTHONPATH=$PYTHONPATH:path/to/reforge/repository
    # or
    pip install -e . # from the reForge repository directory     

Testing the Setup 
-----------------

From the reForge repository directory, you can run the tests with the following command:

.. code-block:: bash

    cd path/to/reforge/repository
    bash run_tests.sh --all

Running the Examples
--------------------

At the moment, the coarse-grained examples can only be run with GROMACS. OpenMM support is in active development. 
Thus, to run the tutorials, you need to have GROMACS installed on your system.
Some basic examples can be found here: `examples <https://github.com/DanYev/cgtools/tree/main/docs/examples>`_, 
and will be updated as the project progresses.

For SOL and PHX Users
---------------------

.. warning::

    Please read this carefully before proceeding! Start with a clean shell and DO NOT activate 
    any environments or interactive sessions unless stated in the instructions. Run the commands EXACTLY 
    as they are in the instructions. Starting an interactive session or running commands 
    with "bash" instead of "source" (.) initiates a new, separate shell process with  
    different environment variables and may break the dependencies.

The first step is the same - we need to clone the repository, 
which will create the directory "reforge" in your current directory and download the GitHub repository.

.. code-block:: bash

    git clone https://github.com/DanYev/reforge.git

Go to the downloaded directory, start a session with enough memory, and run the installation script:

.. code-block:: bash

    cd reforge
    interactive --mem 16G
    . scripts/installation_phx_sol.sh

If the above fails, try requesting more memory, remove the environment, and start over:

.. code-block:: bash

    source deactivate
    mamba env list
    mamba remove -n reforge --all

If the installation was successful, restart the shell (or quit the interactive session by typing *exit*) 
and run the tests. You can find the log in *tests/sl_output.out*

.. code-block:: bash

    cd reforge 
    . scripts/phx_md_load.sh # on PHX
    . scripts/sol_md_load.sh # on SOL
    sbatch run_tests.sh --all

If the above does not work for you, or some of the tests fail, email me at dyangali@asu.edu

