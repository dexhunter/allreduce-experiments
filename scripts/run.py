import torch
import horovod.torch as hvd
import numpy as np
import sys

hvd.init()
torch.cuda.set_device(hvd.local_rank())

# run 10 times take average

TASK = sys.argv[1].split('.')[0]


for i in range(28):
	start = torch.cuda.Event(enable_timing=True)
	end = torch.cuda.Event(enable_timing=True)
	tensor_size = 2**i
	# no cuda
	x = torch.randn(tensor_size, dtype=torch.float).cuda()

	# allreduce op
	start.record()
	reduced = hvd.allreduce(x, average=False)
	end.record()

	torch.cuda.synchronize()

	# print(str(i), start.elapsed_time(end))
	# if hvd.rank() == 0:
# 		record_time.append(start.elapsed_time(end))
	with open("./32gpu/10g/1/"+TASK+'-rank'+str(hvd.rank())+'.txt', "a") as f:
		f.write(str(tensor_size*4)+'\t'+str(start.elapsed_time(end))+'\n')

# if len(record_time) == 28:
# 	with open("./32gpu/10g/mpi-only.txt", "a") as f:
# 		for i in range(28):
# 			f.write(str(record_time[i])+',')
# 
