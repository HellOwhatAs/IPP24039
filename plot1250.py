import random, numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from frame import Frame
from collections import defaultdict
seed=0
lmd=0.6
jobnum=20000

# main
from stragety import single_host,multi_hosts
title="main stragety"
random.seed(seed),np.random.seed(seed)
frame=Frame(single_host,multi_hosts,lmd=lmd)

all_tasks_size=defaultdict(float)
for i in tqdm(range(jobnum),title):
    frame.step()
    
    for i,j in zip(frame.step_assignment,frame.step_j.tasks):
        all_tasks_size[frame.step_host*frame.num_of_cores+i]+=j.size
x,y=[],[]
tmp=max(max(i) for i in frame.cores_on_hosts)
for i in sorted(all_tasks_size.items(),key=lambda i:i[0]):
    x.append(i[0]),y.append(i[1]/tmp)

for i in range(frame.num_of_hosts):
    plt.bar(x[i*frame.num_of_cores:(i+1)*frame.num_of_cores],y[i*frame.num_of_cores:(i+1)*frame.num_of_cores],label="host{}".format(i))
plt.legend()
plt.show()