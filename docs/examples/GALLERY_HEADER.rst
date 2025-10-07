Examples
========

This gallery contains examples of how to use reForge for different molecular dynamics workflows.
Each example is designed to be self-contained and demonstrates specific aspects of the reForge package.

The examples progress from simple concepts to more complex workflows:

1. **Hello World**: Basic CLI interface and job submission patterns
2. **Simple CG Protein**: Complete Go-Martini protein setup workflow  
3. **File Explorer**: Utilities for exploring and visualizing MD outputs
4. **File Management**: Advanced workflow patterns with submit.py

🛠️ **Utility Functions**: The ``utilities.py`` module provides convenient functions for:

   - Opening and displaying file contents with syntax highlighting
   - Listing directory contents and MD outputs systematically  
   - Analyzing PDB structures and generating statistics
   - Creating interactive HTML visualizations of molecular structures
   - Generating comprehensive system summaries with embedded 3D viewers

Requirements for running these examples:

- GROMACS with PLUMED
- Python 3.x with reForge package
- Access to HPC cluster with SLURM (for some examples)
- Web browser for viewing HTML visualizations

Each example includes detailed documentation and can be run independently.
Copy the scripts to your working directory and modify as needed for your specific use case.
The utility functions can be imported and used in your own workflows for enhanced output analysis.

Complete workflows can be found in the `workflows directory <../workflows>`_ of the repository. 