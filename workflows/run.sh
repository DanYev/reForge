#!/bin/bash
#SBATCH --time=0-24:00:00                                                       # upper bound time limit for job to finish d-hh:mm:ss
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
##SBATCH --gres=gpu:1
#SBATCH -o slurm_jobs/output.%A.out
#SBATCH -e slurm_jobs/error.%A.err

# Get the Python script (first argument)
PYSCRIPT="$1"
shift  # Remove the script name from arguments

has_gpus() {
	# Return success (0) iff this job appears to have GPUs allocated.
	# Prefer SLURM variables; fall back to CUDA_VISIBLE_DEVICES.
	# Notes:
	# - SLURM_* may be unset for non-SLURM runs.
	# - CUDA_VISIBLE_DEVICES can be "" or "NoDevFiles" when no GPUs are exposed.
	local v
	for v in "${SLURM_GPUS:-}" "${SLURM_JOB_GPUS:-}" "${SLURM_STEP_GPUS:-}" "${SLURM_GPUS_ON_NODE:-}"; do
		if [[ -n "$v" ]]; then
			return 0
		fi
	done
	if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]]; then
		return 0
	fi
	return 1
}

archive_failure() {
	# Archive logs + the executed Python script when the job fails.
	# Usage: archive_failure <exit_code> <pyscript>
	local rc="${1:-1}"
	local pyscript="${2:-}"

	local date_tag job_tag report_dir stdout_file stderr_file
	date_tag="$(date +%m_%d)"
	job_tag="${SLURM_JOB_NAME:-job}_${SLURM_JOB_ID:-$$}"
	report_dir="$(pwd)/error_logs/${date_tag}/${job_tag}"
	mkdir -p "$report_dir"

	# Copy the executed Python script (wrapper or original).
	if [[ -n "$pyscript" && -f "$pyscript" ]]; then
		cp -f "$pyscript" "$report_dir/" || true

		# If this is an auto-generated wrapper, also copy the frozen script that
		# the wrapper imports (e.g. `import script_20260223_101723 as target_module`).
		local script_dir imported_module imported_script
		script_dir="$(dirname "$pyscript")"
		imported_module="$(grep -m1 -E '^import[[:space:]]+script_[0-9]+' "$pyscript" 2>/dev/null | sed -E 's/^import[[:space:]]+([^[:space:]]+).*/\1/' | tr -d '\r')"
		imported_script=""
		if [[ -n "$imported_module" ]]; then
			imported_script="${script_dir}/${imported_module}.py"
		fi
		if [[ -n "$imported_script" && -f "$imported_script" ]]; then
			cp -f "$imported_script" "$report_dir/" || true
		elif compgen -G "${script_dir}/script_*.py" >/dev/null; then
			# Fallback: copy any frozen scripts sitting next to the wrapper.
			cp -f "${script_dir}/script_"*.py "$report_dir/" || true
		fi
		# Also copy any *.log files that were written alongside the wrapper/script.
		if compgen -G "${script_dir}/*.log" >/dev/null; then
			cp -f "${script_dir}/"*.log "$report_dir/" || true
		fi
	fi

	# SLURM default logs for this job (per #SBATCH -o/-e above).
	stdout_file="$(pwd)/slurm_jobs/output.${SLURM_JOB_ID}.out"
	stderr_file="$(pwd)/slurm_jobs/error.${SLURM_JOB_ID}.err"

	# Copy (archive) if they exist.
	if [[ -n "${SLURM_JOB_ID:-}" && -f "$stdout_file" ]]; then
		cp -f "$stdout_file" "$report_dir/" || true
	fi
	if [[ -n "${SLURM_JOB_ID:-}" && -f "$stderr_file" ]]; then
		cp -f "$stderr_file" "$report_dir/" || true
	fi

	echo "Non-zero exit ($rc); archived to: $report_dir" >&2
}

# Echo all job information to stderr so it appears in the error log file
echo "Job started at: $(date)" >&2
echo "Running script: $PYSCRIPT" >&2
echo "Working directory: $(pwd)" >&2
echo "Arguments: $@" >&2
echo "Environment variables:" >&2
echo "  SLURM_JOB_ID: $SLURM_JOB_ID" >&2
echo "  SLURM_JOB_NAME: $SLURM_JOB_NAME" >&2
echo "  HOSTNAME: $HOSTNAME" >&2

# # --- Temp/runtime directories (OpenMPI/PMIx safety) ---
# # Some clusters have restricted or full /tmp; OpenMPI/PMIx may fail if it can't
# # create its per-job contact files there. Prefer SLURM-provided node-local tmp,
# # else fall back to a local temp folder inside this repo.
# export TMPDIR="${SLURM_TMPDIR:-$(pwd)/tmp/slurm_tmp_${SLURM_JOB_ID:-$$}}"
# mkdir -p "$TMPDIR"
# chmod 700 "$TMPDIR" || true
# # OpenMPI runtime dirs (only read by OpenMPI-based MPI stacks)
# # Use a dedicated subdir so OpenMPI always has a valid parent.
# export OMPI_MCA_orte_tmpdir_base="$TMPDIR/ompi"
# export OMPI_MCA_opal_tmpdir_base="$TMPDIR/ompi"
# mkdir -p "$OMPI_MCA_orte_tmpdir_base"
# chmod 700 "$OMPI_MCA_orte_tmpdir_base" || true
# echo "  TMPDIR: $TMPDIR" >&2

MPS_ENABLED=0
if has_gpus; then
	# Set up CUDA MPS
	MPS_TMPDIR="$(pwd)/tmp/mps_${SLURM_JOB_ID:-$$}"
	export CUDA_MPS_PIPE_DIRECTORY="${MPS_TMPDIR}/nvidia-mps"
	export CUDA_MPS_LOG_DIRECTORY="${MPS_TMPDIR}/nvidia-log"
	mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"

	# Start MPS daemon (if available)
	if command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
		nvidia-cuda-mps-control -d
		MPS_ENABLED=1
	else
		echo "Note: GPUs detected but nvidia-cuda-mps-control not found; skipping MPS." >&2
	fi
else
	echo "No GPUs detected for this job; skipping CUDA MPS setup." >&2
fi
echo "----------------------------------------" >&2

# Run the Python script
python "$PYSCRIPT" "$@"
PY_RC=$?

# Clean up MPS
if [[ "$MPS_ENABLED" -eq 1 ]]; then
	echo quit | nvidia-cuda-mps-control
fi

echo "----------------------------------------" >&2
echo "Job completed at: $(date)" >&2

if [[ "$PY_RC" -ne 0 ]]; then
	archive_failure "$PY_RC" "$PYSCRIPT"
fi

exit "$PY_RC"
