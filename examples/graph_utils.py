"""
Live plotting utilities for IMD streaming workshop.

This module provides a simple interface for creating live-updating plots
during MD simulation streaming with MDAnalysis and IMDv3.
"""

import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import numpy as np


def live_plot(title="Live Analysis", xaxLabel="Time (ps)", yaxLabel="Value", 
              dataLabel=None, figsize=(5, 3), update_interval=1, mode='replace'):
    """
    Create a live-updating plot for streaming MD analysis.
    
    Parameters
    ----------
    title : str
        Plot title
    xaxLabel : str
        X-axis label
    yaxLabel : str  
        Y-axis label
    dataLabel : str
        Label for y-value series
    figsize : tuple
        Figure size (width, height) in inches
    update_interval : int
        Update plot every N frames (1 = every frame, 10 = every 10th frame)
    mode : str
        'replace' to replace data, 'append' to append new data points
    
    Returns
    -------
    dict
        Dictionary with 'fig', 'ax', 'line', 'times', 'values' and 'update' function
    """
    # Setup figure
    plt.ion()
    fig, ax = plt.subplots(figsize=figsize)
    line, = ax.plot([], [], 'b-', linewidth=2, label=dataLabel)
    
    ax.set_xlabel(xaxLabel, fontsize=10)
    ax.set_ylabel(yaxLabel, fontsize=10)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Data storage
    xdata = [[]]
    ydata = [[]]
    frame_counter = [0]  # Use list to allow modification in nested function
    
    def update(x, y):
        """Update the plot with new data"""
        frame_counter[0] += 1
        if mode == 'replace':
            try:
                xdata[0] = x
                ydata[0] = y
            except Exception as e:
                print(f"Error in replacing data: {e}")
        elif mode == 'append':
            try:
                xdata[0].append(x)
                ydata[0].append(y)
            except Exception as e:
                print(f"Error while appending data: {e}")
        else:
            raise ValueError("Mode must be 'replace' or 'append'")

        # Only update display every N frames for performance
        if frame_counter[0] % update_interval == 0:
            line.set_data(xdata[0], ydata[0])
            ax.relim()
            ax.autoscale_view()
            clear_output(wait=True)
            display(fig)
    
    # Return everything in a dict for easy access
    return {
        'fig': fig,
        'ax': ax,
        'line': line,
        'xdata': xdata,
        'ydata': ydata,
        'update': update,
        'close': lambda: plt.ioff()
    }


def live_plot_multi(title="Live Analysis", xaxLabel="Time (ps)", yaxLabel="Value",
                    dataLabels=None, figsize=(5, 3), update_interval=1, mode = 'replace'):
    """
    Create a live-updating plot with multiple y-values for streaming MD analysis.
    
    Parameters
    ----------
    title : str
        Plot title
    xaxLabel : str
        X-axis label
    yaxLabel : str
        Y-axis label
    dataLabels : list of str
        Labels for each y-value series
    figsize : tuple
        Figure size (width, height) in inches
    update_interval : int
        Update plot every N frames
    mode : str
        'replace' to replace data, 'append' to append new data points
    
    Returns
    -------
    dict
        Dictionary with 'fig', 'ax', 'lines', 'times', 'values_list' and 'update' function
    """
    if dataLabels is None:
        dataLabels = ["Value"]
    
    # Setup figure
    plt.ion()
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(dataLabels)))
    lines = []
    for i, (label, color) in enumerate(zip(dataLabels, colors)):
        line, = ax.plot([], [], linewidth=2, label=label, color=color)
        lines.append(line)
    
    ax.set_xlabel(xaxLabel, fontsize=10)
    ax.set_ylabel(yaxLabel, fontsize=10)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Data storage
    xdata = [[]]
    ydata_list = [[] for _ in dataLabels]
    frame_counter = [0]
    
    def update(x,ys):
        """Update the plot with new data."""
        frame_counter[0] += 1
        if mode == 'replace':
            try:
                xdata[0] = x
                for i, val in enumerate(ys):
                    ydata_list[i] = val
            except Exception as e:
                print(f"Error in replacing data: {e}")
        elif mode == 'append':
            try:
                xdata[0].append(x)
                for i, val in enumerate(ys):
                    ydata_list[i].append(val)
            except Exception as e:
                print(f"Error while appending data: {e}")
        else:
            raise ValueError("Mode must be 'replace' or 'append'")
        
        # Only update display every N frames for performance
        if frame_counter[0] % update_interval == 0:
            for i, line in enumerate(lines):
                line.set_data(xdata[0], ydata_list[i])
            ax.relim()
            ax.autoscale_view()
            clear_output(wait=True)
            display(fig)
    
    return {
        'fig': fig,
        'ax': ax,
        'lines': lines,
        'xdata': xdata,
        'ydata_list': ydata_list,
        'update': update,
        'close': lambda: plt.ioff()
    }