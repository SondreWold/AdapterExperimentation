#!/bin/bash
#SBATCH --job-name=thesisWork
#SBATCH --account=ec12
#SBATCH --mail-type=FAIL
#SBATCH --time=30:00:00
#SBATCH --partition=accel --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=10G
#
# by default, request two cores (NumPy may just know how to take
# advantage of them; for larger computations, maybe use between
# six and ten; at some point, we will look at how to run on gpus
#
#SBATCH --ntasks-per-node=2


#
# when running under SLURM control, i.e. as an actual batch job, box in NumPy
# (assuming we stick to the OpenBLAS back-end) to respect our actual allocation
# of cores.
#
if [ -n "${SLURM_JOB_NODELIST}" ]; then
  export OPENBLAS_NUM_THREADS=${SLURM_CPUS_ON_NODE}
fi

# sanity: exit on all errors and disallow unset environment variables
set -o errexit
set -o nounset

# the important bit: unload all current modules (just in case) and load only the necessary ones
module purge
module load Python/3.8.6-GCCcore-10.2.0
export PS1=\$
source venv/bin/activate


# print information (optional)
echo "submission directory: ${SUBMITDIR}"
ulimit -a
module list

# by default, pass on any remaining command-line options
chmod u+x ./run_glue_alt.sh
./run_glue_alt.sh
