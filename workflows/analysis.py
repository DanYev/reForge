import sys
import warnings
import logging
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import MDAnalysis as mda
from sklearn.decomposition import PCA
from sklearn.cluster import BisectingKMeans, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from reforge import io, mdm
from reforge.mdsystem.mdsystem import MDSystem, MDRun
from reforge.utils import clean_dir, logger
import plots

from config import MARTINI, INPDB


def pca_trajs(sysdir, sysname):
    selection = "name CA" # CA for AA or Go-Martini; BB for Martini
    step = 1 # in frames
    mdsys = MDSystem(sysdir, sysname)
    clean_dir(mdsys.datdir, "*")
    tops = io.pull_files(mdsys.mddir, "top.pdb")
    trajs = io.pull_files(mdsys.mddir, "mdc.xtc")
    run_ids = [top.split("/")[-2] for top in tops]
    # Reading 
    logger.info("Reading trajectories")
    u = mda.Universe(tops[0], trajs, in_memory_step=step, ) # in_memory=True)
    ag = u.atoms.select_atoms(selection)
    positions = io.read_positions(u, ag, sample_rate=1, b=0, e=1e9).T
    # PCA
    logger.info("Doing PCA")
    frames = np.arange(len(u.trajectory)) 
    edges = np.cumsum([len(r) for r in u.trajectory.readers])
    traj_ids = np.digitize(frames, edges, right=False)
    pca = PCA(n_components=3)
    x_r = pca.fit_transform(positions) # (n_samples, n_features)
    plot_traj_pca(x_r, 0, 1, traj_ids, run_ids, mdsys, out_tag="runs_pca")
    plot_traj_pca(x_r, 1, 2, traj_ids, run_ids, mdsys, out_tag="runs_pca")
    # Clustering
    cluster(x_r, u, ag, mdsys, n_clusters=2)
    filter_outliers(x_r, u, ag, mdsys)
    logger.info("Done!")


def cluster(data, u, ag, mdsys, n_clusters=2):
    logger.info("Clustering")
    algo = GaussianMixture(n_components=n_clusters, random_state=0, n_init=10)
    # algo = KMeans(n_clusters=n_clusters, random_state=150, n_init=10)
    pred = algo.fit_predict(data)
    labels = []
    for idx, x in enumerate(np.unique(pred)):
        n_samples = np.sum(pred == x)
        label = f"cluster_{idx} with {n_samples} samples"
        labels.append(label)
    plot_traj_pca(data, 0, 1, pred, labels, mdsys, out_tag="clust_pca")
    # plt.scatter(centers[:, 0], centers[:, 1], c="r", s=20)
    for idx, x in enumerate(np.unique(pred)):
        ag.atoms.write(str(mdsys.datdir / f"topology_{idx}.pdb"))
        mask = pred == x
        subset = u.trajectory[mask]
        traj_path = str(mdsys.datdir / f"cluster_{idx}.xtc")
        logger.info(f"Writing cluster %s", idx)
        with mda.Writer(traj_path, ag.n_atoms) as W:
            for ts in subset:   
                W.write(ag) 


def filter_outliers(data, u, ag, mdsys):
    logger.info("Filtering outliers")
    pipe = StandardScaler(with_mean=True, with_std=True)
    Xz = pipe.fit_transform(data)
    ee = EllipticEnvelope(contamination=0.03, support_fraction=0.97,
        assume_centered=True,  random_state=None)
    pred = ee.fit_predict(Xz)               # +1 = inlier (main Gaussian), -1 = outlier
    scores = -ee.score_samples(Xz)          # larger => more outlier-ish
    labels = []
    for idx, x in enumerate(np.unique(pred)):
        n_samples = np.sum(pred == x)
        label = f"cluster_{idx} with {n_samples} samples"
        labels.append(label)
    plot_traj_pca(data, 0, 1, pred, labels, mdsys, out_tag="filtered_pca")
    ag.atoms.write(str(mdsys.datdir / f"filtered.pdb"))
    mask = pred == +1
    subset = u.trajectory[mask]
    traj_path = str(mdsys.datdir / f"filtered.xtc")
    logger.info("Writing filtered cluster")
    with mda.Writer(traj_path, ag.n_atoms) as W:
        for ts in subset:   
            W.write(ag) 


def plot_traj_pca(data, i, j, ids, labels, mdsys, skip=1, alpha=0.3, out_tag="pca",):
    unique_ids = np.unique(ids)
    norm = mcolors.Normalize(vmin=min(ids), vmax=max(ids))
    cmap = plt.get_cmap("viridis")
    plt.figure()
    for tid, label in zip(unique_ids, labels):
        mask = ids == tid
        plt.scatter(data[mask, i][::skip], data[mask, j][::skip],
                    alpha=alpha,
                    color=cmap(norm(tid)),
                    label=label)
    plt.legend()
    plt.xlabel(f"PC{i+1}")
    plt.ylabel(f"PC{j+1}")
    plt.savefig(mdsys.pngdir / f"{out_tag}_{i}{j}.png")
    plt.close()


def clust_cov(sysdir, sysname):
    logger.info("Doing cluster covariance analysis")
    mdsys = MDSystem(sysdir, sysname)
    selection = "name CA"
    clusters = io.pull_files(mdsys.datdir, "cluster*.xtc")
    tops = io.pull_files(mdsys.datdir, "topology*.pdb")
    clusters.append(mdsys.datdir / "filtered.xtc")
    tops.append(mdsys.datdir / "filtered.pdb")
    for idx, (cluster, top) in enumerate(zip(clusters, tops)):
        u = mda.Universe(top, cluster)
        ag = u.atoms.select_atoms(selection)
        dtype = np.float32
        positions = io.read_positions(u, ag, sample_rate=1, b=0, e=1e9, dtype=dtype)
        logger.info("Calculating")
        covmat = mdm.covariance_matrix(positions, dtype=dtype)
        pertmat = mdm.perturbation_matrix_iso(covmat, dtype=dtype)
        dfi_res = mdm.dfi(pertmat)
        idx = 'filt' if cluster == mdsys.datdir / "filtered.xtc" else idx
        np.save(mdsys.datdir / f"cdfi_{idx}_av.npy", dfi_res)
    plots.plot_cluster_dfi(mdsys, tag='cdfi')


def cov_analysis(sysdir, sysname, runname):
    mdrun = MDRun(sysdir, sysname, runname)
    top = mdrun.rundir / "top.pdb"
    traj = mdrun.rundir / "mdc.xtc"
    u = mda.Universe(top, traj, in_memory=False)
    selection = "name CA"
    ag = u.atoms.select_atoms(selection)
    clean_dir(mdrun.covdir, "*npy")
    mdrun.get_covmats(u, ag, sample_rate=1, b=0, e=1e12, n=2, outtag="covmat") 
    mdrun.get_pertmats()
    mdrun.get_dfi(outtag="dfi")
    # mdrun.get_dci(outtag="dci", asym=False)
    # mdrun.get_dci(outtag="asym", asym=True)


def get_averages(sysdir, sysname):
    system = GmxSystem(sysdir, sysname)   
    system.get_mean_sem(pattern="rmsf*.npy")
    system.get_mean_sem(pattern="dfi*.npy")
    system.get_mean_sem(pattern="dci*.npy")
    system.get_mean_sem(pattern="asym*.npy")
    plots.plot_dfi(system, tag='dfi')
    plots.plot_pdfi(system, tag='dfi')


def rdf_analysis(sysdir, sysname, runname, **kwargs):
    system = GmxSystem(sysdir, sysname)
    mdrun = system.initmd(runname)
    rdfndx = os.path.join(system.wdir, "rdf.ndx")
    ions = ["MG", "K", "CL"]
    b = 400000
    for ion in ions:
        mdrun.rdf(clinput=f"BB1\n {ion}\n", n=rdfndx, o=f"rms_analysis/rdf_{ion}.xvg", 
            b=b, rmax=10, bin=0.01, **kwargs)

    
def rms_analysis(sysdir, sysname, runname, **kwargs):
    kwargs.setdefault("b",  450000) # in ps
    kwargs.setdefault("dt", 150) # in ps
    kwargs.setdefault("e", 7500000) # in ps
    mdrun = GmxRun(sysdir, sysname, runname)
    clean_dir(mdrun.rmsdir, "*npy")
    mdrun.rmsf(clinput=f"2\n 2\n", s=mdrun.str, f=mdrun.trj, n=mdrun.sysndx, fit="yes", res="yes", **kwargs) 
    mdrun.get_rmsf_by_chain(**kwargs)
    mdrun.rmsd(clinput=f"2\n 2\n", s=mdrun.str, f=mdrun.trj, n=mdrun.sysndx, fit="rot+trans", **kwargs)
    mdrun.get_rmsd_by_chain(b=0, **kwargs)
    # u = mda.Universe(mdrun.str, mdrun.trj, in_memory=True)
    # ag = u.atoms.select_atoms("name BB or name BB2")
    # positions = io.read_positions(u, ag, b=500000, e=3500000, sample_rate=1)
    # mdm.calc_and_save_rmsf(positions, outdir=mdrun.rmsdir, n=20)
      
    
def overlap(sysdir, sysname, **kwargs):
    system = GmxSystem(sysdir, sysname)
    run1 = system.initmd("mdrun_2")
    run2 = system.initmd("mdrun_4")
    run3 = system.initmd("mdrun_5")
    v1 = os.path.join(run1.covdir, "eigenvec.trr")
    v2 = os.path.join(run2.covdir, "eigenvec.trr")
    v3 = os.path.join(run3.covdir, "eigenvec.trr")
    run1.anaeig(v=v1, v2=v2, over="overlap_1.xvg", **kwargs)
    run1.anaeig(v=v2, v2=v3, over="overlap_2.xvg", **kwargs)
    run1.anaeig(v=v3, v2=v1, over="overlap_3.xvg", **kwargs)


def tdlrt_analysis(sysdir, sysname, runname):
    mdrun = GmxRun(sysdir, sysname, runname)
    # CCF params FRAMEDT=20 ps
    b = 0
    e = 100000
    sample_rate = 1
    ntmax = 1000 # how many frames to save
    fname = "corr_pv.npy"
    corr_file = os.path.join(mdrun.lrtdir, fname)
    # CALC CCF
    u = mda.Universe(mdrun.str, mdrun.trj, in_memory=True)
    ag = u.atoms
    positions = io.read_positions(u, ag, sample_rate=sample_rate, b=b, e=e) 
    velocities = io.read_velocities(u, ag, sample_rate=sample_rate, b=b, e=e)
    corr = lrt.ccf(positions, velocities, ntmax=ntmax, n=5, mode="gpu", center=True)
    np.save(corr_file, corr)


def get_td_averages(sysdir, sysname):
    system = GmxSystem(sysdir, sysname)  
    system.get_td_averages("pertmat*.npy", loop=True)


def test(sysdir, sysname, runname, **kwargs):    
    print('passed', file=sys.stderr)

