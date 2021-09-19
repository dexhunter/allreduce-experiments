#!/bin/bash
MPI_HOME=/home/esetstore/.local/openmpi-4.0.1
# 100GbIB
$MPI_HOME/bin/mpirun --prefix $MPI_HOME --oversubscribe -np 32 -H gpu1:4,gpu2:4,gpu3:4,gpu4:4,gpu5:4,gpu6:4,gpu7:4,gpu8:4 -bind-to none -map-by slot -mca pml ob1 -mca btl openib -mca btl_openib_allow_ib 1 \
-x LD_LIBRARY_PATH="/usr/local/cuda-10.1/lib64:/home/esetstore/downloads/nccl_2.6.4-1+cuda10.1_x86_64/lib:$LD_LIBRARY_PATH" \
-x NCCL_DEBUG=INFO \
-x NCCL_TREE_THRESHOLD=0 \
-x CUDA_VISIBLE_DEVICES=0,1,2,3 \
-x NCCL_SOCKET_IFNAME=ib0 \
./build/all_reduce_perf -b 4 -e 512M -g 1 -f 2 > nccl-32gpus/100g-stepexp2-2.txt
# ./build/all_reduce_perf -b 1M -e 512M -i 2000000 -g 1 > nccl-32gpus/ring-100g-step2m.txt
#./build/broadcast_perf -b 8 -e 512M -f 2 -g 1
#./build/reduce_perf -b 8 -e 1024M -f 2 -g 1

# 10GbE
export ETH_INTERFACE=enp136s0f0,enp137s0f0
export ETH_MPI_BTC_TCP_IF_INCLUDE=192.168.0.1/24
$MPI_HOME/bin/mpirun --oversubscribe --prefix $MPI_HOME -np 32 -H gpu1:4,gpu2:4,gpu3:4,gpu4:4,gpu5:4,gpu6:4,gpu7:4,gpu8:4 -bind-to none -map-by slot \
-mca pml ob1 -mca btl ^openib \
-mca btl_tcp_if_include ${ETH_MPI_BTC_TCP_IF_INCLUDE} \
-x LD_LIBRARY_PATH="/usr/local/cuda-10.1/lib64:/home/esetstore/downloads/nccl_2.6.4-1+cuda10.1_x86_64/lib:$LD_LIBRARY_PATH" \
-x NCCL_DEBUG=INFO  \
-x NCCL_TREE_THRESHOLD=0 \
-x CUDA_VISIBLE_DEVICES=0,1,2,3 \
-x NCCL_SOCKET_IFNAME=${ETH_INTERFACE} \
-x NCCL_IB_DISABLE=1 \
./build/all_reduce_perf -b 4 -e 512M -g 1 -f 2 > nccl-32gpus/10g-stepexp2-2.txt

echo "-------------------------------------------"

export NCCL_DEBUG=INFO
export LD_LIBRARY_PATH=/home/esetstore/downloads/nccl_2.6.4-1+cuda10.1_x86_64/lib:$LD_LIBRARY_PATH
# ./build/all_reduce_perf -b 1M -e 512M -f 2 -g 4 > nccl-32gpus/test-10g-step2m-2.txt
