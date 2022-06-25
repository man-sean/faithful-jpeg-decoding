#!/usr/bin/env bash

CPUS=$1
GPUS=$2
PORT=${PORT:-$3}  #PORT=${PORT:-4321}
CONFIG=$4

## usage
#if [ $# -lt 3 ] ;then
#    echo "usage:"
#    echo "./scripts/slurm_train.sh [number of gpu] [path to option file]"
#    exit
#fi
#
#PYTHONPATH="$(dirname $0)/..:${PYTHONPATH}" \
#python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#    basicsr/train.py -opt $CONFIG --launcher pytorch ${@:3}

#case "$DEBUG_CUDA" in
#    ON) CUDA_LAUNCH_BLOCKING=1 ;;
#    *) CUDA_LAUNCH_BLOCKING=0 ;;
#esac

PYTHONPATH="./:${PYTHONPATH}" \
GLOG_vmodule=MemcachedClient=-1 \
#CUDA_LAUNCH_BLOCKING=1 \ # for debugging
#[ "$DEBUG_CUDA" == "ON" ] && CUDA_LAUNCH_BLOCKING=1 \ || CUDA_LAUNCH_BLOCKING=0 \
#WANDB_MODE=offline \
OMP_NUM_THREADS=$(($CPUS/$GPUS)) \
srun -p cactus -A cactus --mpi=pmi2 --job-name=JPEG --gres=gpu:$GPUS --ntasks=1 --ntasks-per-node=1 --cpus-per-task=$CPUS --kill-on-bad-exit=1 \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
basicsr/train.py -opt $CONFIG --launcher="pytorch" ${@:5}