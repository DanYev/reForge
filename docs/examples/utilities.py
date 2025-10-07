#!/usr/bin/env python
"""
Example Utilities
=================

Utility functions for file handling, output listing, and PDB visualization in HTML format.

This module provides convenient functions for:
- Opening and displaying file contents
- Listing directory contents and outputs
- Creating HTML visualizations of PDB structures
- Generating interactive plots and reports

Requirements:
    - Python 3.x
    - matplotlib (for plotting)
    - MDAnalysis (for structure analysis, optional)
    - py3Dmol (for 3D visualization, optional)

Author: DY
"""

import os
import sys
from pathlib import Path
from typing import Union, List, Optional, Dict, Any
import subprocess
import tempfile
import webbrowser

def open_file(filepath: Union[str, Path], max_lines: int = 100, show_line_numbers: bool = True) -> None:
    """
    Open and display the contents of a file with optional line numbers.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the file to display
    max_lines : int, default 100
        Maximum number of lines to display (prevents overwhelming output)
    show_line_numbers : bool, default True
        Whether to show line numbers
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        return
    
    print(f"📄 Contents of {filepath}")
    print("=" * 60)
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        total_lines = len(lines)
        display_lines = min(max_lines, total_lines)
        
        for i, line in enumerate(lines[:display_lines]):
            if show_line_numbers:
                print(f"{i+1:4d}: {line.rstrip()}")
            else:
                print(line.rstrip())
        
        if total_lines > max_lines:
            print(f"\n... ({total_lines - max_lines} more lines not shown)")
            print(f"💡 Use max_lines={total_lines} to see all lines")
            
    except UnicodeDecodeError:
        print("⚠️  File appears to be binary - showing first few bytes:")
        with open(filepath, 'rb') as f:
            content = f.read(1000)
            print(content)
    except Exception as e:
        print(f"❌ Error reading file: {e}")


def list_directory(dirpath: Union[str, Path], pattern: str = "*", 
                  show_details: bool = True, max_files: int = 50) -> None:
    """
    List directory contents with details and filtering.
    
    Parameters
    ----------
    dirpath : str or Path
        Directory to list
    pattern : str, default "*"
        Glob pattern to filter files (e.g., "*.pdb", "*.log")
    show_details : bool, default True
        Show file sizes and modification times  
    max_files : int, default 50
        Maximum number of files to display
    """
    dirpath = Path(dirpath)
    
    if not dirpath.exists():
        print(f"❌ Directory not found: {dirpath}")
        return
    
    if not dirpath.is_dir():
        print(f"❌ Not a directory: {dirpath}")
        return
    
    print(f"📁 Contents of {dirpath}")
    if pattern != "*":
        print(f"🔍 Filter: {pattern}")
    print("=" * 80)
    
    try:
        files = list(dirpath.glob(pattern))
        files.sort()
        
        if not files:
            print("📭 No files found")
            return
        
        display_files = files[:max_files]
        
        for filepath in display_files:
            if show_details:
                stat = filepath.stat()
                size = format_file_size(stat.st_size)
                mtime = format_timestamp(stat.st_mtime)
                file_type = "📁" if filepath.is_dir() else "📄"
                print(f"{file_type} {filepath.name:<40} {size:>10} {mtime}")
            else:
                file_type = "📁" if filepath.is_dir() else "📄"
                print(f"{file_type} {filepath.name}")
        
        if len(files) > max_files:
            print(f"\n... ({len(files) - max_files} more files not shown)")
            
    except Exception as e:
        print(f"❌ Error listing directory: {e}")


def list_outputs(sysdir: Union[str, Path], sysname: str, patterns: List[str] = None) -> Dict[str, List[Path]]:
    """
    List common MD simulation output files in a systematic way.
    
    Parameters
    ----------
    sysdir : str or Path
        System directory (e.g., 'systems')
    sysname : str
        System name (e.g., 'test')
    patterns : list of str, optional
        File patterns to search for. Defaults to common MD outputs.
    
    Returns
    -------
    dict
        Dictionary mapping file categories to lists of found files
    """
    if patterns is None:
        patterns = {
            'PDB Structures': ['*.pdb'],
            'GROMACS Files': ['*.gro', '*.top', '*.itp', '*.ndx'],
            'Trajectories': ['*.xtc', '*.trr', '*.dcd'],
            'Logs': ['*.log', '*.out', '*.err'],
            'Energy Files': ['*.edr', '*.xvg'],
            'MDP Files': ['*.mdp'],
            'Scripts': ['*.sh', '*.py']
        }
    
    sysdir = Path(sysdir)
    system_path = sysdir / sysname
    
    if not system_path.exists():
        print(f"❌ System directory not found: {system_path}")
        return {}
    
    print(f"🔍 Output files for system: {sysname}")
    print(f"📂 Location: {system_path}")
    print("=" * 80)
    
    results = {}
    
    for category, file_patterns in patterns.items():
        found_files = []
        for pattern in file_patterns:
            found_files.extend(system_path.glob(f"**/{pattern}"))
        
        found_files = sorted(set(found_files))  # Remove duplicates and sort
        results[category] = found_files
        
        if found_files:
            print(f"\n📋 {category}:")
            for filepath in found_files:
                rel_path = filepath.relative_to(system_path)
                size = format_file_size(filepath.stat().st_size)
                print(f"   📄 {rel_path} ({size})")
        else:
            print(f"\n📋 {category}: None found")
    
    return results


def create_pdb_html(pdb_file: Union[str, Path], output_html: Optional[Union[str, Path]] = None,
                   style: str = "cartoon", color: str = "spectrum", width: int = 800, 
                   height: int = 600, show_surface: bool = False) -> str:
    """
    Create an HTML file with interactive 3D visualization of a PDB structure.
    
    Parameters
    ----------
    pdb_file : str or Path
        Path to the PDB file
    output_html : str or Path, optional
        Output HTML file path. If None, creates one based on PDB filename.
    style : str, default "cartoon"
        Visualization style: "cartoon", "stick", "sphere", "line"
    color : str, default "spectrum"
        Coloring scheme: "spectrum", "chain", "residue", "atom"
    width : int, default 800
        Viewer width in pixels
    height : int, default 600
        Viewer height in pixels
    show_surface : bool, default False
        Whether to show molecular surface
    
    Returns
    -------
    str
        Path to the created HTML file
    """
    pdb_file = Path(pdb_file)
    
    if not pdb_file.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_file}")
    
    if output_html is None:
        output_html = pdb_file.with_suffix('.html')
    else:
        output_html = Path(output_html)
    
    # Read PDB content
    with open(pdb_file, 'r') as f:
        pdb_content = f.read()
    
    # Create HTML content with py3Dmol viewer
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PDB Viewer - {pdb_file.name}</title>
    <script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: {width + 40}px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .viewer {{
            border: 2px solid #ddd;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .controls {{
            margin: 10px 0;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 5px;
        }}
        button {{
            margin: 5px;
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            background: #007bff;
            color: white;
            cursor: pointer;
        }}
        button:hover {{
            background: #0056b3;
        }}
        .info {{
            background: #e9ecef;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧬 PDB Structure Viewer</h1>
        <div class="info">
            <strong>File:</strong> {pdb_file.name}<br>
            <strong>Size:</strong> {format_file_size(pdb_file.stat().st_size)}<br>
            <strong>Location:</strong> {pdb_file.absolute()}
        </div>
        
        <div class="controls">
            <strong>Style:</strong>
            <button onclick="setStyle('cartoon')">Cartoon</button>
            <button onclick="setStyle('stick')">Stick</button>
            <button onclick="setStyle('sphere')">Sphere</button>
            <button onclick="setStyle('line')">Line</button>
            <br>
            <strong>Color:</strong>
            <button onclick="setColor('spectrum')">Spectrum</button>
            <button onclick="setColor('chain')">Chain</button>
            <button onclick="setColor('residue')">Residue</button>
            <button onclick="setColor('atom')">Atom</button>
            <br>
            <button onclick="toggleSurface()">Toggle Surface</button>
            <button onclick="viewer.zoomTo(); viewer.render();">Zoom to Fit</button>
            <button onclick="viewer.spin(!viewer.isSpinning());">Toggle Spin</button>
        </div>
        
        <div id="viewer" class="viewer"></div>
        
        <div class="info">
            <h3>🎮 Interaction Guide</h3>
            <ul>
                <li><strong>Rotate:</strong> Click and drag</li>
                <li><strong>Zoom:</strong> Mouse wheel or pinch</li>
                <li><strong>Pan:</strong> Right-click and drag</li>
                <li><strong>Reset:</strong> Use "Zoom to Fit" button</li>
            </ul>
        </div>
    </div>

    <script>
        // Initialize 3Dmol viewer
        let viewer = $3Dmol.createViewer("viewer", {{
            width: {width},
            height: {height},
            backgroundColor: "white"
        }});
        
        // PDB data
        let pdbData = `{pdb_content}`;
        
        // Add model to viewer
        viewer.addModel(pdbData, "pdb");
        
        // Set initial style
        viewer.setStyle({{}}, {{{style}: {{colorscheme: "{color}"}}}});
        {"viewer.addSurface($3Dmol.VDW, {opacity: 0.7, color: 'white'});" if show_surface else ""}
        
        // Center and render
        viewer.zoomTo();
        viewer.render();
        
        // Control functions
        function setStyle(style) {{
            viewer.setStyle({{}}, {{}});
            let styleObj = {{}};
            styleObj[style] = {{colorscheme: getCurrentColor()}};
            viewer.setStyle({{}}, styleObj);
            viewer.render();
        }}
        
        function setColor(color) {{
            let currentStyle = getCurrentStyle();
            let styleObj = {{}};
            styleObj[currentStyle] = {{colorscheme: color}};
            viewer.setStyle({{}}, styleObj);
            viewer.render();
        }}
        
        function getCurrentStyle() {{
            // This is a simplified approach - in practice you'd track current style
            return "{style}";
        }}
        
        function getCurrentColor() {{
            // This is a simplified approach - in practice you'd track current color
            return "{color}";
        }}
        
        let surfaceVisible = {str(show_surface).lower()};
        function toggleSurface() {{
            if (surfaceVisible) {{
                viewer.removeAllSurfaces();
                surfaceVisible = false;
            }} else {{
                viewer.addSurface($3Dmol.VDW, {{opacity: 0.7, color: 'white'}});
                surfaceVisible = true;
            }}
            viewer.render();
        }}
        
        // Auto-spin option
        // viewer.spin(true);
    </script>
</body>
</html>"""
    
    # Write HTML file
    with open(output_html, 'w') as f:
        f.write(html_content)
    
    print(f"✅ Created HTML visualization: {output_html}")
    print(f"🌐 Open in browser: file://{output_html.absolute()}")
    
    return str(output_html)


def create_summary_html(sysdir: Union[str, Path], sysname: str, 
                       output_html: Optional[Union[str, Path]] = None) -> str:
    """
    Create a comprehensive HTML summary of a system with all outputs and visualizations.
    
    Parameters
    ----------
    sysdir : str or Path
        System directory
    sysname : str  
        System name
    output_html : str or Path, optional
        Output HTML file path
        
    Returns
    -------
    str
        Path to the created HTML summary
    """
    sysdir = Path(sysdir)
    system_path = sysdir / sysname
    
    if output_html is None:
        output_html = system_path / f"{sysname}_summary.html"
    else:
        output_html = Path(output_html)
    
    # Get all outputs
    outputs = list_outputs(sysdir, sysname)
    
    # Find PDB files for visualization
    pdb_files = outputs.get('PDB Structures', [])
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>System Summary - {sysname}</title>
    <script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .section {{
            padding: 20px;
            margin: 10px;
            border-radius: 10px;
            background: #f8f9fa;
            border-left: 4px solid #007bff;
        }}
        .file-list {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
            margin: 15px 0;
        }}
        .file-item {{
            background: white;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #ddd;
            transition: all 0.3s ease;
        }}
        .file-item:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}
        .viewer-container {{
            margin: 20px 0;
            text-align: center;
        }}
        .viewer {{
            border: 2px solid #ddd;
            border-radius: 10px;
            margin: 10px auto;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{ margin-top: 0; }}
        .badge {{
            background: #007bff;
            color: white;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            margin-left: 10px;
        }}
        .no-files {{
            color: #6c757d;
            font-style: italic;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧬 System Summary: {sysname}</h1>
            <p>📂 Location: {system_path}</p>
        </div>
"""
    
    # Add file listings
    for category, files in outputs.items():
        html_content += f"""
        <div class="section">
            <h2>{category} <span class="badge">{len(files)}</span></h2>
"""
        if files:
            html_content += '<div class="file-list">'
            for filepath in files:
                rel_path = filepath.relative_to(system_path)
                size = format_file_size(filepath.stat().st_size)
                html_content += f"""
                <div class="file-item">
                    <strong>📄 {filepath.name}</strong><br>
                    <small>📁 {rel_path.parent}/</small><br>
                    <small>📏 {size}</small>
                </div>
"""
            html_content += '</div>'
        else:
            html_content += '<p class="no-files">No files found</p>'
        
        html_content += '</div>'
    
    # Add PDB visualizations
    if pdb_files:
        html_content += """
        <div class="section">
            <h2>🔬 Structure Visualizations</h2>
"""
        for i, pdb_file in enumerate(pdb_files[:3]):  # Limit to first 3 PDBs
            pdb_content = ""
            try:
                with open(pdb_file, 'r') as f:
                    pdb_content = f.read().replace('`', '\\`')  # Escape backticks
            except:
                continue
                
            html_content += f"""
            <div class="viewer-container">
                <h3>📄 {pdb_file.name}</h3>
                <div id="viewer_{i}" class="viewer"></div>
            </div>
"""
    
    html_content += """
        </div>
    </div>

    <script>
"""
    
    # Add JavaScript for PDB viewers
    if pdb_files:
        for i, pdb_file in enumerate(pdb_files[:3]):
            try:
                with open(pdb_file, 'r') as f:
                    pdb_content = f.read().replace('\\', '\\\\').replace('`', '\\`').replace('$', '\\$')
                
                html_content += f"""
        // Viewer {i}
        let viewer_{i} = $3Dmol.createViewer("viewer_{i}", {{
            width: 600,
            height: 400,
            backgroundColor: "white"
        }});
        
        let pdbData_{i} = `{pdb_content}`;
        viewer_{i}.addModel(pdbData_{i}, "pdb");
        viewer_{i}.setStyle({{}}, {{cartoon: {{colorscheme: "spectrum"}}}});
        viewer_{i}.zoomTo();
        viewer_{i}.render();
"""
            except:
                continue
    
    html_content += """
    </script>
</body>
</html>"""
    
    # Write HTML file
    with open(output_html, 'w') as f:
        f.write(html_content)
    
    print(f"✅ Created system summary: {output_html}")
    print(f"🌐 Open in browser: file://{output_html.absolute()}")
    
    return str(output_html)


def open_in_browser(filepath: Union[str, Path]) -> None:
    """
    Open an HTML file in the default web browser.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the HTML file
    """
    filepath = Path(filepath)
    if filepath.exists() and filepath.suffix.lower() == '.html':
        webbrowser.open(f"file://{filepath.absolute()}")
        print(f"🌐 Opened in browser: {filepath}")
    else:
        print(f"❌ HTML file not found: {filepath}")


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def format_timestamp(timestamp: float) -> str:
    """Format timestamp in human-readable format."""
    import datetime
    dt = datetime.datetime.fromtimestamp(timestamp)
    return dt.strftime("%Y-%m-%d %H:%M")


def quick_analysis(pdb_file: Union[str, Path]) -> Dict[str, Any]:
    """
    Perform quick analysis of a PDB file and return summary statistics.
    
    Parameters
    ----------
    pdb_file : str or Path
        Path to PDB file
        
    Returns
    -------
    dict
        Analysis results including atom counts, residue counts, etc.
    """
    pdb_file = Path(pdb_file)
    
    if not pdb_file.exists():
        return {"error": f"File not found: {pdb_file}"}
    
    stats = {
        "filename": pdb_file.name,
        "size": format_file_size(pdb_file.stat().st_size),
        "atoms": 0,
        "residues": set(),
        "chains": set(),
        "hetero_atoms": 0,
        "waters": 0
    }
    
    try:
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')):
                    stats["atoms"] += 1
                    chain_id = line[21:22].strip()
                    if chain_id:
                        stats["chains"].add(chain_id)
                    
                    res_name = line[17:20].strip()
                    res_num = line[22:26].strip()
                    stats["residues"].add(f"{chain_id}:{res_name}{res_num}")
                    
                    if line.startswith('HETATM'):
                        stats["hetero_atoms"] += 1
                        if res_name in ('HOH', 'WAT', 'TIP', 'SOL'):
                            stats["waters"] += 1
        
        stats["residues"] = len(stats["residues"])  
        stats["chains"] = len(stats["chains"])
        
    except Exception as e:
        stats["error"] = str(e)
    
    return stats


# Example usage and demo functions
def demo_utilities():
    """Demonstrate the utility functions with example data."""
    print("🧪 reForge Example Utilities Demo")
    print("=" * 50)
    
    # Check if we have example systems
    systems_dir = Path("systems")
    if not systems_dir.exists():
        print("⚠️  No 'systems' directory found. Run go_protein.py first!")
        return
    
    # Find available systems
    system_dirs = [d for d in systems_dir.iterdir() if d.is_dir()]
    if not system_dirs:
        print("⚠️  No system directories found. Run go_protein.py first!")
        return
    
    # Use the first system found
    system_name = system_dirs[0].name
    print(f"📂 Using system: {system_name}")
    
    # Demo 1: List directory contents
    print("\n1️⃣ Directory Listing Demo:")
    list_directory(systems_dir / system_name, pattern="*.pdb")
    
    # Demo 2: List all outputs  
    print("\n2️⃣ Output Listing Demo:")
    outputs = list_outputs("systems", system_name)
    
    # Demo 3: Show file contents
    print("\n3️⃣ File Content Demo:")
    pdb_files = outputs.get('PDB Structures', [])
    if pdb_files:
        open_file(pdb_files[0], max_lines=20)
    
    # Demo 4: Quick PDB analysis
    print("\n4️⃣ PDB Analysis Demo:")
    if pdb_files:
        stats = quick_analysis(pdb_files[0])
        print("📊 Structure Statistics:")
        for key, value in stats.items():
            if key != "error":
                print(f"   {key}: {value}")
    
    # Demo 5: Create HTML visualizations
    print("\n5️⃣ HTML Visualization Demo:")
    if pdb_files:
        # Create individual PDB visualization
        html_file = create_pdb_html(pdb_files[0])
        
        # Create system summary
        summary_html = create_summary_html("systems", system_name)
        
        print(f"💡 Open these files in your browser to see the visualizations!")


if __name__ == "__main__":
    demo_utilities()
