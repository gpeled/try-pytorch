#!/usr/bin/env python

from __future__ import print_function
import sys
import datetime
sys.path.append('/home/gpeled/workspace/pytorch/python3-pytorch/lib/python3.5/site-packages/')
#print ('\n'.join(sys.path))

import torch


print('Date now: %s' % datetime.datetime.now())
print("creating tensors. Time: %s"% datetime.datetime.now())
d1=1024
d2=1024
x = torch.rand(d1, d2).cuda()
y = torch.rand(d1, d2).cuda()

start_time=datetime.datetime.now()
print("Starting math. Time: %s"% start_time)
#result = torch.Tensor(d1, d2)
for i in range(100000):
    y.add_(x)
    #y=result
    #z=x.std()
    if not(i % 100):
        if (i % 4000):
            print(".",end="")
            sys.stdout.flush()
        else:
            print("\n%d " %i,end="")

end_time = datetime.datetime.now()
print("\nDONE. Time: %s" % end_time)
print("Time spent in math: %s" % (end_time-start_time))


