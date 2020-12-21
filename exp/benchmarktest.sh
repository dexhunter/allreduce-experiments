mpirun --prefix <location of openmpi> -np <number of processes> -hostfile <dns of servers> --mca coll_tuned_allreduce_algorithm <number of algorithm performed> <location to omb>/osu-micro-benchmarks-5.7/mpi/collective/osu_allreduce

# example

# ring algorithm
mpirun --prefix <location of openmpi> -np 2 -hostfile <dns of servers> --mca coll_tuned_allreduce_algorithm 4 <location to omb>/osu-micro-benchmarks-5.7/mpi/collective/osu_allreduce
mpirun --prefix <location of openmpi> -np 4 -hostfile <dns of servers> --mca coll_tuned_allreduce_algorithm 4 <location to omb>/osu-micro-benchmarks-5.7/mpi/collective/osu_allreduce
mpirun --prefix <location of openmpi> -np 8 -hostfile <dns of servers> --mca coll_tuned_allreduce_algorithm 4 <location to omb>/osu-micro-benchmarks-5.7/mpi/collective/osu_allreduce
mpirun --prefix <location of openmpi> -np 16 -hostfile <dns of servers> --mca coll_tuned_allreduce_algorithm 4 <location to omb>/osu-micro-benchmarks-5.7/mpi/collective/osu_allreduce
mpirun --prefix <location of openmpi> -np 32 -hostfile <dns of servers> --mca coll_tuned_allreduce_algorithm 4 <location to omb>/osu-micro-benchmarks-5.7/mpi/collective/osu_allreduce

