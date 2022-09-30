#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE        #Do not propagate environment
#SBATCH --get-user-env=L     #Replicate login environment
  
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=C90_M3_P100_G50     #Set the job name to "JobExample1"
#SBATCH --time=02:00:00            #Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=50                #Request 400 task
#SBATCH --ntasks-per-node=25        #Request 10 task/core per node
#SBATCH --mem=20480M                #Request 204800MB (20GB) per node
#SBATCH --output=Example1Out.%j    #Send stdout/err to "Example1Out.[jobID]"

##OPTIONAL JOB SPECIFICATIONS
##SBATCH --account=132774976391             #Set billing account to 122774970664

ml GCC/10.3.0
ml CMake/3.18.0

export PATH=$SCRATCH/EnergyPlus_WithMatActuator/Products:$PATH

mpirun python MPC_NSF_Impl_HPRC.py
