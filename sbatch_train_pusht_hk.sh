#!/bin/bash 
#SBATCH -J diffusion_policy_pusht_hk
#SBATCH -p gpux
#SBATCH -n 1       # Number of tasks
#SBATCH -c 20      # Number of cores per task
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time 72:00:00
#SBATCH --comment=LightVLA
#SBATCH -o /public/home/group_xudong/yimingwu/project/MyProjects/LightVLA/examples/diffusion_policy/jobs/diffusion_policy_pusht_hk/std.out.%j
#SBATCH -e /public/home/group_xudong/yimingwu/project/MyProjects/LightVLA/examples/diffusion_policy/jobs/diffusion_policy_pusht_hk/std.err.%j


#########################################################
### Slurm scripts for Sugon Portal_5.0 of BASE
### Version 1.0    |  2019-09-09  |  created by Zhang Guoliang
### Version 1.0.1  |  2020-11-24  |  modified by Zhang Guoliang
#########################################################

### Get parameters from GUI

# Remove references to portal files that don't exist
# MIDFILE_DIR=/public/home/group_xudong/yimingwu/project/MyProjects/LightVLA/examples/mdt/jobs/mdt_calvin_hk/.portal
# source $MIDFILE_DIR/job_portal.var
# source $MIDFILE_DIR/job_interface.var

### Set basic var   ### MARK_slurm2pbs

WORK_DIR=/public/home/group_xudong/yimingwu/project/MyProjects/LightVLA/examples/mdt/jobs/mdt_calvin_hk
JOB_NAME=mdt_calvin_hk
JOBID=$SLURM_JOB_ID
NP=$SLURM_NPROCS
NNODE=`srun hostname | sort | uniq | wc -l`

LOG_FILE=$WORK_DIR/job_${JOB_NAME}_${JOBID}.log
HOST_FILE=$WORK_DIR/job_${JOB_NAME}_${JOBID}_${NP}c_${NNODE}n.ma 

# Create the directory if it doesn't exist
mkdir -p $WORK_DIR

# Generate host file
srun hostname | sort | uniq -c | awk '{print $2":"$1}' > $HOST_FILE

# Log basic job information
echo -e "The start time is: `date +"%Y-%m-%d %H:%M:%S"` \n" | tee -a $LOG_FILE 
echo -e "My job ID is: $JOBID \n" | tee -a $LOG_FILE  
echo -e "The total cores is: $NP \n" | tee -a $LOG_FILE 
echo -e "The hosts is: \n" | tee -a $LOG_FILE
cat $HOST_FILE | tee -a $LOG_FILE
echo -e "\n" | tee -a $LOG_FILE 

# Run the application
echo "Running the application..." | tee -a $LOG_FILE

# 设置分布式训练所需的环境变量
export MASTER_ADDR=$(hostname -i)
export MASTER_PORT=$(( 10000 + $RANDOM % 20000 ))
export WORLD_SIZE=1

# Change to the project directory
cd /public/home/group_xudong/yimingwu/project/MyProjects/LightVLA/examples/diffusion_policy

# Properly activate conda environment
# Use the full path to conda and source the conda.sh file
source /public/home/group_xudong/yimingwu/miniconda3/etc/profile.d/conda.sh
conda activate robodiff

# Run the training script with full path
srun --ntasks=1 --ntasks-per-node=1 --cpus-per-task=20 --gres=gpu:1 \
    python train.py --config-dir=. --config-name=train_diffusion_light_transformer_hybrid_workspace.yaml task=pusht_image hydra.run.dir='data/outputs/prune_by_learning/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
    # python train.py --config-dir=. --config-name=train_diffusion_transformer_hybrid_workspace.yaml task=pusht_image hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

echo "The end time is: `date +"%Y-%m-%d %H:%M:%S"` | tee -a $LOG_FILE"
