mpirun --prefix <location of openmpi> -np <number of processes> -hostfile <dns of servers> --mca coll_tuned_allreduce_algorithm <number of algorithm performed> <location to omb>/osu-micro-benchmarks-5.7/mpi/collective/osu_allreduce

# example

# ring algorithm
# message size up to 1GB
# memroy limit up to 8GB (7GB available)
# force using eth0 interface
# debug mode
mpirun --prefix <location of openmpi> -np <number of processes> -hostfile <dns of servers> --mca btl_tcp_if_include 192.168.0.1/24 --mca orte_base_help_aggregate 0 --mca coll_tuned_allreduce_algorithm 4 <location to omb>/mpi/collective/osu_allreduce -m 1073741824 -M 7550136320
