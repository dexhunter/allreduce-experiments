# allreduce-experiments

## Open MPI

Implementation of different allreduce algorithms

* [x] Reduce + Broadcast
* [ ] All-to-all Communication
* [x] Recursive Doubling
* [ ] Recursive Halving
* [x] Rabenseifner's Algorithm
* [ ] Butterfly
* [ ] 2D Mesh
* [ ] 2D Torus
* [ ] 3D Torus
* [x] Ring
* [ ] Binary Double Tree


## NCCL Test

## Horovod

* NCCL_ALLREDUCE
* MPI_ALLREDUCE
* NCCL_REDUCESCATTER+MPI_ALLREDUCE+NCCL_ALLGATHER
* NCCL_REDUCE+MPI_ALLREDUCE+NCCL_BCAST
* NCCL_REDUCESCATTER+NCCL_ALLREDUCE+NCCL_ALLGATHER
* NCCL_REDUCE+NCCL_ALLREDUCE+NCCL_BCAST

* more on the [blog](https://blog.dex.moe/tutorial/2021/06/08/how-to-write-custom-allreduce-operation.html)