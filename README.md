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
* [MPI_ALLREDUCE](./impl/horovod/MPI_ALLREDUCE.cc)
* [NCCL_REDUCESCATTER+MPI_ALLREDUCE+NCCL_ALLGATHER](./impl/horovod/NCCL_REDUCESCATTER+MPI_ALLREDUCE+NCCL_ALLGATHER.cc)
* [NCCL_REDUCE+MPI_ALLREDUCE+NCCL_BCAST](./impl/horovod/NCCL_REDUCE+MPI_ALLREDUCE+NCCL_BCAST.cc)
* [NCCL_REDUCESCATTER+NCCL_ALLREDUCE+NCCL_ALLGATHER](./impl/horovod/NCCL_REDUCESCATTER+NCCL_ALLREDUCE+NCCL_ALLGATHERE.cc)
* [NCCL_REDUCE+NCCL_ALLREDUCE+NCCL_BCAST](./impl/horovod/NCCL_REDUCE+NCCL_ALLREDUCE+NCCL_BCAST.cc)

* more on the [blog](https://blog.dex.moe/tutorial/2021/06/08/how-to-write-custom-allreduce-operation.html)