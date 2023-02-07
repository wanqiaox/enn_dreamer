#!/bin/bash

sbatch <<EOT
#!/bin/bash

#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --time=7-00:00:00
#SBATCH --mem=42G
#SBATCH -J `basename "$1"`
#SBATCH --output=slurm-%j.out

# only use the following on partition with GPUs
#SBATCH --gres=gpu:"$2"

source /sailhome/${USER}/.bashrc
conda activate dreamer

export PYTHONPATH=$PWD:$PYTHONPATH

# python dreamer.py --logdir ./logdir/walker_walk-p2e --configs defaults dmc
python dreamer.py --logdir ./logdir/$1-greedy --configs defaults dmc --task dmc_$1
# python dreamer.py --logdir ./logdir/reward-$1-p2e --configs defaults dmc p2e_reward --task dmc_$1

# done
echo "Done"
EOT
