import inspect
import multiprocessing as mp
import os
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import MDAnalysis as mda
from MDAnalysis.analysis import rms
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import BisectingKMeans, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from reforge import io, mdm
from reforge.mdsystem.mdsystem import MDSystem, MDRun
from reforge.utils import clean_dir, get_logger
import plots

logger = get_logger(__name__)


SELECTION = "protein" 
TRJEXT = 'trr' # 'xtc' or 'trr'

################################################################################
### PCA/Clustering ###
################################################################################
def pca_trajs(sysdir, sysname):
    selection = SELECTION # CA for AA or Go-Martini; BB for Martini
    step = 1 # in frames
    mdsys = MDSystem(sysdir, sysname)
    clean_dir(mdsys.datdir, "*")
    tops = io.pull_files(mdsys.mddir, "topology.pdb")
    trajs = io.pull_files(mdsys.mddir, f"samples.{TRJEXT}")
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
    _plot_traj_pca(x_r, 0, 1, traj_ids, run_ids, mdsys, out_tag="runs_pca")
    _plot_traj_pca(x_r, 1, 2, traj_ids, run_ids, mdsys, out_tag="runs_pca")
    # Clustering
    _cluster(x_r, u, ag, mdsys, n_clusters=2)
    _filter_outliers(x_r, u, ag, mdsys)
    logger.info("Done!")


def _cluster(data, u, ag, mdsys, n_clusters=2):
    logger.info("Clustering")
    algo = GaussianMixture(n_components=n_clusters, random_state=0, n_init=10)
    # algo = KMeans(n_clusters=n_clusters, random_state=150, n_init=10)
    pred = algo.fit_predict(data)
    labels = []
    for idx, x in enumerate(np.unique(pred)):
        n_samples = np.sum(pred == x)
        label = f"cluster_{idx} with {n_samples} samples"
        labels.append(label)
    _plot_traj_pca(data, 0, 1, pred, labels, mdsys, out_tag="clust_pca")
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


def _filter_outliers(data, u, ag, mdsys):
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
    _plot_traj_pca(data, 0, 1, pred, labels, mdsys, out_tag="filtered_pca")
    ag.atoms.write(str(mdsys.datdir / f"filtered.pdb"))
    mask = pred == +1
    subset = u.trajectory[mask]
    traj_path = str(mdsys.datdir / f"filtered.xtc")
    logger.info("Writing filtered cluster")
    with mda.Writer(traj_path, ag.n_atoms) as W:
        for ts in subset:   
            W.write(ag) 


def _plot_traj_pca(data, i, j, ids, labels, mdsys, skip=1, alpha=0.3, out_tag="pca",):
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
    selection = SELECTION
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

################################################################################
### DFI/DCI ###
################################################################################

def cov_analysis(sysdir, sysname, runname):
    mdrun = MDRun(sysdir, sysname, runname)
    mdrun.covdir.mkdir(exist_ok=True, parents=True)
    top = mdrun.rundir / "topology.pdb"
    traj = mdrun.rundir / f"samples.{TRJEXT}"
    u = mda.Universe(top, traj, in_memory=False)
    selection = "name CA"
    ag = u.atoms.select_atoms(selection)
    clean_dir(mdrun.covdir, "*npy")
    mdrun.get_covmats(u, ag, sample_rate=1, b=0, e=1e12, n=2, outtag="covmat") 
    mdrun.get_pertmats()
    mdrun.get_dfi(outtag="dfi")
    mdrun.get_dci(outtag="dci", asym=False)
    mdrun.get_dci(outtag="asym", asym=True)


def get_means_sems(sysdir, sysname):
    system = MDSystem(sysdir, sysname)   
    system.get_mean_sem(pattern="rmsf*.npy")
    system.get_mean_sem(pattern="dfi*.npy")
    system.get_mean_sem(pattern="dci*.npy")
    system.get_mean_sem(pattern="asym*.npy")
    plots.plot_dfi(system, tag='dfi')
    plots.plot_pdfi(system, tag='dfi')

################################################################################
### RMSD/RMSF Analysis ###
################################################################################

def rms_analysis(sysdir, sysname, runname, selection=SELECTION, step=1):
    mdsys = MDSystem(sysdir, sysname)
    mdrun = MDRun(sysdir, sysname, runname)
    rmsdir = mdrun.rmsdir
    rmsdir.mkdir(exist_ok=True)    
    top = mdrun.rundir / "topology.pdb"
    traj = mdrun.rundir / f"samples.{TRJEXT}"
    # Load trajectory
    u = mda.Universe(str(top), str(traj))
    atoms = u.select_atoms(selection)  
    # Calculate RMSD
    logger.info(f'Calculating RMSD and RMSF for selection: {selection}')
    rmsd_analysis = rms.RMSD(atoms, reference=atoms, select=selection)
    rmsd_analysis.run(step=step)
    # Calculate RMSF
    rmsf_analysis = rms.RMSF(atoms)
    rmsf_analysis.run(step=step)
    # Get residue IDs
    residue_ids = np.array([atom.resid for atom in atoms])
    # Save arrays
    np.save(rmsdir / "rmsd_values.npy", rmsd_analysis.rmsd[:, 2])  # RMSD values (in angstroms)
    np.save(rmsdir / "rmsd_times.npy", rmsd_analysis.rmsd[:, 1])   # Time values (in ps)
    np.save(rmsdir / "rmsf_values.npy", rmsf_analysis.rmsf)        # RMSF values (in angstroms)
    np.save(rmsdir / "residue_ids.npy", residue_ids)               # Residue IDs
    logger.info(f"Saved RMSD and RMSF data to {rmsdir}")
    # Plots
    logger.info("Generating RMSD and RMSF plots")
    plots.plot_rmsd(mdsys)
    plots.plot_rmsf(mdsys)
    
################################################################################
### TDLRT ###
################################################################################

def tdlrt_analysis(sysdir, sysname, runname):
    mdrun = MDRun(sysdir, sysname, runname)
    mdrun.lrtdir.mkdir(exist_ok=True, parents=True)
    ps_path = mdrun.rundir / "positions.npy"
    vs_path = mdrun.rundir / "velocities.npy"
    if (ps_path.exists() and vs_path.exists()):
        logger.info("Loading positions and velocities from %s", mdrun.rundir)
        ps = np.load(ps_path)
        vs = np.load(vs_path)
    else:
        traj = mdrun.rundir / f"samples.{TRJEXT}"
        top = mdrun.rundir / "topology.pdb"
        u = mda.Universe(top, traj)
        ps = io.read_positions(u, u.atoms) # (n_atoms*3, nframes)
        vs = io.read_velocities(u, u.atoms) # (n_atoms*3, nframes)
    ps = ps - ps[:, 0][..., None]
    # CCF calculations
    adict = {'pp': (ps, ps), } 
    for key, item in adict.items(): # DT = TSTEP * NOUT
        v1, v2 = item
        corr = mdm.ccf(v1, v2, ntmax=100, n=1, mode='gpu', center=False, dtype=np.float32, buffer_c=0.9) # falls back on cpu if no cuda
        corr_file = mdrun.lrtdir / f'ccfs_{key}.npy'
        np.save(corr_file, corr)    
        logger.info("Saved CCFs to %s", corr_file)


def get_averages(sysdir, sysname, pattern="ccfs_pp*.npy", dtype=None):
    """Calculate average arrays across files matching pattern."""
    mdsys = MDSystem(sysdir, sysname)
    nprocs = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
    logger.info("Number of available processors: %s", nprocs)
    files = io.pull_files(sysdir, pattern)[::1]
    if not files:
        logger.info('Could not find files matching given pattern: %s. Maybe you forgot "*"?', pattern)
        return
    logger.info("Found %d files, starting processing: %s", len(files), files[0])
    # Discover minimal common shape (fast, uses mmap to avoid loading full arrays)
    shapes = []
    for f in files:
        try:
            arr = np.load(f, mmap_mode='r')
            if dtype is None:
                dtype = arr.dtype
            shapes.append(arr.shape)
        except Exception as e:
            logger.warning("Could not read shape for %s: %s", f, e)
    if not shapes:
        logger.info('No readable files found for pattern: %s', pattern)
        return
    min_shape = tuple(min(s[i] for s in shapes) for i in range(len(shapes[0])))
    logger.info('Running parallel get_averages with %d processes', nprocs)
    # split files into roughly equal batches
    batches = [files[i::nprocs] for i in range(nprocs)]
    work = [(batch, min_shape) for batch in batches if batch]
    with mp.Pool(processes=len(work)) as pool:
        results = pool.map(_process_batch, work)
    total_sum = np.zeros(min_shape, dtype=dtype)
    total_count = 0
    for local_sum, local_count in results:
        total_sum += local_sum
        total_count += local_count
    average = total_sum / total_count
    outdir = mdsys.datdir
    outdir.mkdir(exist_ok=True, parents=True)
    out_file = outdir / f"{pattern.split('*')[0]}_av.npy"
    np.save(out_file, average)
    logger.info("Saved averages to %s", out_file)


def _process_batch(args, dtype=np.float32):
    """Worker: load assigned files, crop to min_shape and return local sum and count."""
    files, min_shape = args
    s = tuple(slice(0, s) for s in min_shape)
    local_sum = np.zeros(min_shape, dtype=dtype)
    local_count = 0
    for f in files:
        logger.info("Processing %s", f)
        try:
            arr = np.load(f)
        except Exception as e:
            logger.warning("Could not load %s: %s", f, e)
            continue
        local_sum += arr[s]
        local_count += 1
        del arr
    return local_sum, local_count

################################################################################
### Bioemu ###
################################################################################

def sample_emu(sysdir, sysname, runname):
    from bioemu.sample import main as sample
    mdrun = MDRun(sysdir, sysname, runname)
    mdrun.prepare_files()
    sequence = _pdb_to_seq(mdrun.sysdir / INPDB)
    sample(sequence=sequence, num_samples=1000, batch_size_100=20, output_dir=mdrun.rundir)


def initiate_systems_from_emu(*args):
    logger.info("Preparing directories from EMU samples")
    emu_dir = Path("systems") / "emu"
    newsys_dir = Path("systems") / "1btl_nve"
    samples = emu_dir / "samples.xtc"
    top = emu_dir / "topology.pdb"
    u = mda.Universe(top, samples)
    step = 10  # every 10 frames
    for i, ts in enumerate(u.trajectory[1::step]):
        idx = i 
        outdir = newsys_dir / f"sample_{idx:03d}"
        outdir.mkdir(parents=True, exist_ok=True)
        outpdb = outdir / "sample.pdb"
        with mda.Writer(outpdb, u.atoms.n_atoms) as W:
            W.write(u.atoms)
        logger.info(f"Saved initial structure {i} to {outpdb}")


def _pdb_to_seq(pdb):
    u = mda.Universe(pdb)
    protein = u.select_atoms("protein")
    seq = "".join(res.resname for res in protein.residues)  # three-letter codes
    seq_oneletter = "".join(mda.lib.util.convert_aa_code(res.resname) for res in protein.residues)
    return seq_oneletter





if __name__ == "__main__":
    from reforge.cli import run_command
    run_command()