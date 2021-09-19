#!/bin/bash

export HOROVOD_NCCL_HOME=/home/esetstore/downloads/nccl_2.6.4-1+cuda10.1_x86_64 
export NCCL_DEBUG=INFO
export PATH=/home/esetstore/.local/openmpi-4.0.1/bin:$PATH
export LD_LIBRARY_PATH=/home/esetstore/downloads/nccl_2.6.4-1+cuda10.1_x86_64/lib:$LD_LIBRARY_PATH

TASK=nccl-only

mpirun --prefix /home/esetstore/.local/openmpi-4.0.1 -np 32 -H gpu1:4,gpu2:4,gpu3:4,gpu4:4,gpu5:4,gpu6:4,gpu7:4,gpu8:4 --mca pml ob1 --mca btl ^openib --mca orte_base_help_aggregate 0 --mca btl_tcp_if_include 192.168.0.1/24 -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -x HOROVOD_NCCL_HOME -x HOROVOD_TIMELINE=./32gpu/logs/10g/$TASK.json -x NCCL_SOCKET_IFNAME=enp136s0f0,enp137s0f0 -x NCCL_IB_DISABLE=1 -x HOROVOD_CACHE_CAPACITY=0 -x CUDA_VISIBLE_DEVICES=0,1,2,3 python3 run.py $TASK

mpirun --prefix /home/esetstore/.local/openmpi-4.0.1 -np 32 -H gpu1:4,gpu2:4,gpu3:4,gpu4:4,gpu5:4,gpu6:4,gpu7:4,gpu8:4 --mca pml ob1 --mca btl openib,vader,self --mca btl_openib_allow_ib 1 --mca orte_base_help_aggregate 0 --mca btl_tcp_if_include ib0 --mca btl_openib_want_fork_support 1 -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -x HOROVOD_NCCL_HOME -x HOROVOD_TIMELINE=./32gpu/logs/100g/$TASK.json -x NCCL_IB_DISABLE=0 -x NCCL_SOCKET_IFNAME=ib0 -x HOROVOD_CACHE_CAPACITY=0 -x CUDA_VISIBLE_DEVICES=0,1,2,3 python3 100g-run.py $TASK

