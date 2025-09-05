import os
import numpy as np
import pandas as pd
import sys
from reforge import io, mdm
from reforge.mdsystem import gmxmd
from reforge.plotting import *
from reforge.utils import logger


def pull_data(datdir, metric):
    files = io.pull_files(datdir, metric)
    datas = [np.load(f) for f in files if '_av' in f]
    errs = [np.load(f) for f in files if '_err' in f]
    fnames = [f.split("/")[-1] for f in files if '_av' in f]
    return datas, errs, fnames


def set_bfactors_by_residue(in_pdb, bfactors, out_pdb=None):
    atoms = io.pdb2atomlist(in_pdb)
    residues = atoms.residues
    for idx, residue in enumerate(residues):
        for atom in residue:
            atom.bfactor = bfactors[idx]
    if out_pdb:
        atoms.write_pdb(out_pdb)
    return atoms


def set_bfactors_by_atom(in_pdb, bfactors, out_pdb=None):
    atoms = io.pdb2atomlist(in_pdb)
    for idx, atom in enumerate(atoms):
        atom.bfactor = bfactors[idx]
    if out_pdb:
        atoms.write_pdb(out_pdb)
    return atoms


def set_ax_parameters(ax, xlabel=None, ylabel=None, axtitle=None, loc=None):
    """
    ax - matplotlib ax object
    """
    # Set axis labels and title with larger font sizes
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(axtitle, fontsize=16)
    # Customize tick parameters
    ax.tick_params(axis='both', which='major', labelsize=14, direction='in', length=5, width=1.5)
    # Increase spine width for a bolder look
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    # Add a legend with custom font size and no frame
    legend = ax.legend(fontsize=14, frameon=False, loc=loc)
    # Optionally, add gridlines
    ax.grid(True, linestyle='--', alpha=0.5)


def plot_dfi(system, tag='dfi'):
    datas, errs, fnames = pull_data(system.datdir, f"{tag}*")
    xs = [np.arange(len(data))+26 for data in datas]
    labels = [f.split(".")[0] for f in fnames]
    params = [{'lw':2, 'label':label} for label in labels]
    # Plotting
    fig, ax = init_figure(grid=(1, 1), axsize=(12, 5))
    make_errorbar(ax, xs, datas, errs, params, alpha=0.7)
    # make_plot(ax, xs, datas, params)
    set_ax_parameters(ax, xlabel='Residue', ylabel='DFI', loc='upper right')
    plot_figure(fig, ax, figname=system.sysname.upper(), figpath=system.pngdir / f"{tag}.png",)


def plot_pdfi(system, tag='dfi'):
    datas, errs, fnames = pull_data(system.datdir, f"{tag}*")
    xs = [np.arange(len(data))+26 for data in datas]
    datas = [mdm.percentile(data) for data in datas]
    labels = [f.split(".")[0] for f in fnames]
    params = [{'lw':2, 'label':label} for label in labels]
    # Plotting
    fig, ax = init_figure(grid=(1, 1), axsize=(12, 5))
    make_plot(ax, xs, datas, params)
    set_ax_parameters(ax, xlabel='Residue', ylabel='%DFI', loc='lower right')
    plot_figure(fig, ax, figname=system.sysname.upper(), figpath=system.pngdir / f"p{tag}.png",)


def plot_cluster_dfi(system, tag='cdfi'):
    datas, errs, fnames = pull_data(system.datdir, f"{tag}*")
    xs = [np.arange(len(data))+26 for data in datas]
    pdatas = [mdm.percentile(data) for data in datas]
    labels = [f.split(".")[0] for f in fnames]
    colors = ['silver', 'grey', 'black']
    lws = [1, 1, 2]
    params = [{'color':c, 'label':l, 'lw':lw, } for c, l, lw in zip(colors, labels, lws)]
    # Plotting DFI
    fig, ax = init_figure(grid=(1, 1), axsize=(12, 5))
    make_plot(ax, xs, datas, params)
    set_ax_parameters(ax, xlabel='Residue', ylabel='DFI', loc='upper right')
    plot_figure(fig, ax, figname=system.sysname.upper(), figpath=system.pngdir / f"{tag}.png",)
    # Plotting PDFI
    fig, ax = init_figure(grid=(1, 1), axsize=(12, 5))
    make_plot(ax, xs, pdatas, params)
    set_ax_parameters(ax, xlabel='Residue', ylabel='%DFI', loc='lower right')
    plot_figure(fig, ax, figname=system.sysname.upper(), figpath=system.pngdir / f"p{tag}.png",)


def plot_rmsf(system):
    # Pulling data
    datas, errs = pull_data(system.datdir, 'crmsf_B*')
    xs = [np.arange(len(data)) for data in datas]
    datas = [data*10 for data in datas]
    errs = [err*10 for err in errs]
    params = [{'lw':2} for data in datas]
    # Plotting
    fig, ax = init_figure(grid=(1, 1), axsize=(12, 5))
    make_errorbar(ax, xs, datas, errs, params, alpha=0.7)
    set_ax_parameters(ax, xlabel='Residue', ylabel='RMSF (Angstrom)')
    plot_figure(fig, ax, figname=system.sysname.upper(), figpath='png/rmsf.png',)


def plot_rmsd(system):
    # Pulling data
    files = io.pull_files(system.mddir, 'rmsd*npy')
    datas = [np.load(file) for file in files]
    labels = [file.split('/')[-3] for file in files]
    xs = [data[0]*1e-3 for data in datas]
    ys = [data[1]*10 for data in datas]
    params = [{'lw':2, 'label':label} for label in labels]
    # Plotting
    fig, ax = init_figure(grid=(1, 1), axsize=(12, 5))
    make_plot(ax, xs, ys, params)
    set_ax_parameters(ax, xlabel='Time (ns)', ylabel='RMSD (Angstrom)')
    plot_figure(fig, ax, figname=system.sysname.upper() , figpath=system.pngdir / 'rmsd.png',)


def plot_dci(system):
    # Pulling data
    datas, errs = pull_data(system.datdir, 'pertmat*')
    param = {'lw':2}
    datas = [data for data in datas]
    data = datas[0]
    data = mdm.dci(data)
    # Plotting
    fig, ax = init_figure(grid=(1, 1), axsize=(12, 12))
    make_heatmap(ax, data, cmap='bwr', interpolation=None, vmin=0, vmax=2)
    set_ax_parameters(ax, xlabel='Residue', ylabel='Residue')
    plot_figure(fig, ax, figname='DCI', figpath='png/dci.png',)


def plot_asym(system):
    # Pulling data
    datas, errs = pull_data(system.datdir, 'asym*')
    param = {'lw':2}
    datas = [data for data in datas]
    data = datas[0]
    # Plotting
    fig, ax = init_figure(grid=(1, 1), axsize=(12, 12))
    make_heatmap(ax, data, cmap='bwr', interpolation=None, vmin=-1, vmax=1)
    set_ax_parameters(ax, xlabel='Residue', ylabel='Residue')
    plot_figure(fig, ax, figname='DCI asymmetry', figpath='png/asym.png',)


def make_pdb(system, label, factor=None):
    data = np.load(os.path.join(system.datdir, f'{label}_av.npy'))
    err = np.load(os.path.join(system.datdir, f'{label}_err.npy'))
    if factor:
        data *= factor
        err *= factor
    data_pdb = os.path.join(system.pngdir, f'{label}.pdb')
    err_pdb = os.path.join(system.pngdir, f'{label}_err.pdb')
    set_bfactors_by_residue(system.inpdb, data, data_pdb)
    set_bfactors_by_residue(system.inpdb, err, err_pdb)


def make_cg_pdb(system, label, factor=None):
    data = np.load(os.path.join(system.datdir, f'{label}_av.npy'))
    err = np.load(os.path.join(system.datdir, f'{label}_err.npy'))
    if factor:
        data *= factor
        err *= factor
    data_pdb = os.path.join(system.pngdir, f'{label}.pdb')
    err_pdb = os.path.join(system.pngdir, f'{label}_err.pdb')
    set_bfactors_by_atom(system.root / 'mdci.pdb', data, data_pdb)
    set_bfactors_by_atom(system.root / 'mdci.pdb', err, err_pdb)


def make_enm_pdb(system, label, factor=None):
    data = np.load(os.path.join(system.datdir, f'{label}_enm.npy'))
    if factor:
        data *= factor
    data_pdb = os.path.join(system.pngdir, f'enm_{label}.pdb')
    set_bfactors_by_residue(system.inpdb, data, data_pdb)


def make_delta_pdb(system_1, system_2, label, out_name, filter=True, factor=None):
    logger.info('Making Delta PDB')
    data_1 = np.load(os.path.join(system_1.datdir, f'{label}_av.npy'))
    err_1 = np.load(os.path.join(system_1.datdir, f'{label}_err.npy'))
    data_2 = np.load(os.path.join(system_2.datdir, f'{label}_av.npy'))
    err_2 = np.load(os.path.join(system_2.datdir, f'{label}_err.npy'))  
    if factor:
        data_1 *= factor
        err_1 *= factor
        data_2 *= factor
        err_2 *= factor
    data = data_1 - data_2
    err = np.sqrt(err_1**2 + err_2**2)
    if filter:
        mask = np.abs(data) < 2.0 * err
        data[mask] = 0
    data_pdb = os.path.join('systems', 'pdb', out_name + '.pdb')
    err_pdb = os.path.join('systems', 'pdb', out_name + '_err.pdb')
    set_bfactors_by_residue(system_1.inpdb, data, data_pdb)
    set_bfactors_by_residue(system_1.inpdb, err, err_pdb) 
    logger.info('Saved Delta PDB to %s', data_pdb)


def make_delta_cg_pdb(system_1, system_2, label, out_name, factor=None):
    data_1 = np.load(os.path.join(system_1.datdir, f'{label}_av.npy'))
    err_1 = np.load(os.path.join(system_1.datdir, f'{label}_err.npy'))
    data_2 = np.load(os.path.join(system_2.datdir, f'{label}_av.npy'))
    err_2 = np.load(os.path.join(system_2.datdir, f'{label}_err.npy'))  
    if factor:
        data_1 *= factor
        err_1 *= factor
        data_2 *= factor
        err_2 *= factor
    data = data_1 - data_2
    err = np.sqrt(err_1**2 + err_2**2)
    data_pdb = os.path.join('systems', 'pdb', out_name + '.pdb')
    err_pdb = os.path.join('systems', 'pdb', out_name + '_err.pdb')
    set_bfactors_by_atom(system.root / 'mdci.pdb', data, data_pdb)
    set_bfactors_by_atom(system.root / 'mdci.pdb', err, err_pdb)

def rmsf_pdb(system):
    logger.info(f'Making RMSF PDB')
    make_cg_pdb(system, 'rmsf', factor=10)


def dfi_pdb(system):
    logger.info(f'Making DFI PDB')
    make_pdb(system, 'dfi')


def dci_pdbs(system):
    # chains = ['W']
    chains = system.chains
    for chain in chains:
        logger.info(f'Making DCI {chain} PDB')
        label = f'gdci_{chain}'
        make_pdb(system, label)


def pocket_dci_pdbs(system):
    pockets = ['ptc']
    for chain in pockets:
        logger.info(f'Making DCI {chain} PDB')
        label = f'gdci_{chain}'
        make_pdb(system, label)


def runs_metric(system, metric):
    files = io.pull_files(system.mddir, metric)
    files = [f for f in files if '.npy' in f]
    datas = [np.load(file) for file in files]
    xs = [np.arange(len(data)) for data in datas]
    datas = [data for data in datas]
    params = [{'lw':2, 'label':fname} for data, fname in zip(datas, files)]
    # Plotting
    fig, ax = init_figure(grid=(1, 1), axsize=(12, 5))
    make_plot(ax, xs, datas, params)
    set_ax_parameters(ax, xlabel='Residue', ylabel='RMSF (Angstrom)')
    plot_figure(fig, ax, figname=system.sysname.upper(), figpath='png/metric.png',)


def make_hpf_pdb(system):
    data_1 = np.load(os.path.join(system.datdir, f'gdci_T_av.npy'))
    data_2 = np.load(os.path.join(system.datdir, f'gdci_W_av.npy'))
    data_3 = np.load(os.path.join(system.datdir, f'gdci_Q_av.npy'))
    data = (data_1 + data_2) / 2.0
    data_pdb = os.path.join(system.pngdir, f'hpf.pdb')
    set_bfactors_by_residue(system.inpdb, data, data_pdb)


if __name__ == '__main__':
    sysdir = 'systems_kras' 
    system = gmxmd.GmxSystem(sysdir, 'go_gmx')
    # plot_cluster_dfi(system,)
    plot_dfi(system, tag="dfi")
    plot_pdfi(system, tag="dfi")
    # plot_ggdci(system)
    # plot_dci(system)
    # plot_asym(system)
    # plot_rmsf(system)
    # plot_rmsd(system)
    # # # PDBs
    # rmsf_pdb(system)
    # dfi_pdb(system)
    # dci_pdbs(system)
    # pocket_dci_pdbs(system)
    # make_enm_pdb(system, label='dfi')
    # make_hpf_pdb(system)
    # # Deltas
    # variant_1 = 'mgh'
    # variant_2 = 'wt'
    # label = 'dfi'
    # system_1 = gmxmd.GmxSystem(sysdir, f'ribosome_{variant_1}')
    # system_2 = gmxmd.GmxSystem(sysdir, f'ribosome_{variant_2}')
    # make_delta_pdb(system_1, system_2, label=label, out_name=f'{label}_{variant_1}_{variant_2}', filter=True)
    # # make_delta_cg_pdb(system_1, system_2, label='rmsf', out_name=f'drmsf_{variant_1}_{variant_2}')
    # # runs_metric(system, 'rmsf*')





