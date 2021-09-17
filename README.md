# allreduce-experiments

## Open MPI

Implementation of different allreduce algorithms

* basic_linear
* nonoverlapping
* recursive_doubling
* ring
* segmented_ring
* rabenseifner

## NCCL Test

* ring

## Horovod

* [NCCL_ALLREDUCE](./impl/horovod/NCCL_ALLREDUCE.cc)
* MPI_ALLREDUCE
* NCCL_REDUCESCATTER+MPI_ALLREDUCE+NCCL_ALLGATHER
* NCCL_REDUCE+MPI_ALLREDUCE+NCCL_BCAST
* NCCL_REDUCESCATTER+NCCL_ALLREDUCE+NCCL_ALLGATHER
* NCCL_REDUCE+NCCL_ALLREDUCE+NCCL_BCAST

* more on the [blog](https://blog.dex.moe/tutorial/2021/06/08/how-to-write-custom-allreduce-operation.html)