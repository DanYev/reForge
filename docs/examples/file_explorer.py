#!/usr/bin/env python
"""
File Explorer Example
=====================

Exploring MD outputs with utilities

This example demonstrates how to use the utilities.py module to explore
and visualize molecular dynamics simulation outputs.

Requirements:
    - Python 3.x
    - reForge package
    - A completed simulation (run go_protein.py first)

Author: DY
"""

# Import our utility functions
from utilities import (
    open_file, list_directory, list_outputs, 
    quick_analysis, create_pdb_html, create_summary_html,
    demo_utilities, open_in_browser
)
from pathlib import Path

#%%
# First, let's see what systems we have available
print("🔍 Exploring Available Systems")
print("=" * 50)

systems_dir = Path("systems")
if systems_dir.exists():
    print(f"📁 Systems directory found: {systems_dir}")
    list_directory(systems_dir, show_details=True)
else:
    print("❌ No systems directory found!")
    print("💡 Tip: Run go_protein.py first to create a system")
    exit()

#%%
# Find available systems and select one
system_dirs = [d for d in systems_dir.iterdir() if d.is_dir()]
if not system_dirs:
    print("❌ No system subdirectories found!")
    print("💡 Tip: Run go_protein.py first to create a system")
    exit()

# Use the first available system
selected_system = system_dirs[0].name
print(f"\n🎯 Selected system: {selected_system}")

#%%
# Explore the system contents in detail
print(f"\n📂 Detailed exploration of system '{selected_system}':")
list_directory(systems_dir / selected_system, pattern="*", show_details=True)

#%%
# Look specifically for PDB files
print(f"\n🧬 PDB structures in '{selected_system}':")
list_directory(systems_dir / selected_system, pattern="*.pdb", show_details=True)

#%%
# Show contents of a key file (e.g., topology)
print(f"\n📄 Examining topology file contents:")
top_files = list((systems_dir / selected_system).glob("*.top"))
if top_files:
    open_file(top_files[0], max_lines=30)
else:
    print("   No .top files found")

#%%
# Get comprehensive output listing
print(f"\n📋 Comprehensive output analysis for '{selected_system}':")
outputs = list_outputs("systems", selected_system)

#%%
# Analyze PDB structures
pdb_files = outputs.get('PDB Structures', [])
if pdb_files:
    print(f"\n🔬 Structure Analysis:")
    for pdb_file in pdb_files[:3]:  # Analyze first 3 PDB files
        print(f"\n📊 Analysis of {pdb_file.name}:")
        stats = quick_analysis(pdb_file)
        for key, value in stats.items():
            if key != "error":
                print(f"   {key}: {value}")
else:
    print("\n⚠️  No PDB files found for analysis")

#%%
# Create HTML visualizations
print(f"\n🌐 Creating interactive visualizations...")

# Create individual structure visualizations
html_files = []
for i, pdb_file in enumerate(pdb_files[:2]):  # Limit to first 2 for demo
    html_file = create_pdb_html(
        pdb_file, 
        output_html=pdb_file.with_suffix('.html'),
        style="cartoon" if i == 0 else "stick",
        color="spectrum" if i == 0 else "chain"
    )
    html_files.append(html_file)

# Create comprehensive system summary
if pdb_files:
    summary_file = create_summary_html("systems", selected_system)
    html_files.append(summary_file)

#%%
# Display what was created
print(f"\n✅ Generated {len(html_files)} HTML visualization files:")
for html_file in html_files:
    html_path = Path(html_file)
    print(f"   🌐 {html_path.name}")
    print(f"      file://{html_path.absolute()}")

print(f"\n💡 Instructions:")
print(f"   1. Copy any of the file:// URLs above")
print(f"   2. Paste in your web browser")  
print(f"   3. Explore the interactive 3D structures!")

#%%
# Demonstrate file content exploration
print(f"\n📖 File Content Exploration:")
log_files = outputs.get('Logs', [])
if log_files:
    print(f"\nShowing excerpt from log file: {log_files[0].name}")
    open_file(log_files[0], max_lines=20)

script_files = outputs.get('Scripts', [])  
if script_files:
    print(f"\nShowing script file: {script_files[0].name}")
    open_file(script_files[0], max_lines=15)

#%%
# Summary and next steps
print(f"\n🎉 File Exploration Complete!")
print(f"=" * 50)
print(f"📊 System: {selected_system}")
print(f"📁 Location: {systems_dir / selected_system}")
print(f"🌐 HTML files: {len(html_files)} created")
print(f"🔬 PDB files: {len(pdb_files)} analyzed")

if html_files:
    print(f"\n🚀 Next Steps:")
    print(f"   • Open the HTML files in your browser")
    print(f"   • Use the interactive controls to explore structures")  
    print(f"   • Check the system summary for complete overview")

#%%
# Optional: Run the full demo
print(f"\n🧪 Running Full Utilities Demo:")
demo_utilities()
