# modify the osu_latency for step size
# loc: /<PATH TO OSU MICRO BENCNMARKS LIB>/mpi/pt2pt/osu_latency.c
# line 99
# change to
# size = (size ? size + <DESIRED AMOUNT> : 1)) 

mpirun --prefix /home/t716/shshi/share/local/openmpi-4.0.1 -np 2 -hostfile hostlatency /nfs_home/dxxu/omb-5.7-no-cuda/mpi/pt2pt/osu_latency --message-size 4:4000 --mem-limit 7550136320 > 22to23-step32.txt
# mpirun --prefix /home/t716/shshi/share/local/openmpi-4.0.1 -np 2 -hostfile hostlatency /nfs_home/dxxu/omb-5.7-no-cuda/mpi/pt2pt/osu_latency --message-size 4000:4000000 --mem-limit 7550136320 > 22to23-step1000.txt
# mpirun --prefix /home/t716/shshi/share/local/openmpi-4.0.1 -np 2 -hostfile hostlatency /nfs_home/dxxu/omb-5.7-no-cuda/mpi/pt2pt/osu_latency --message-size 4000000:512000000 --mem-limit 7550136320 > 22to23-step1000000.txt
