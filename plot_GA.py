import random, numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from frame import Frame
from collections import defaultdict
seed=0
lmd=0.6
jobnum=2000

# main
from stragety import single_host,multi_hosts
title="main stragety"
random.seed(seed),np.random.seed(seed)
frame=Frame(single_host,multi_hosts,lmd=lmd)
r=[]
all_tasks_size=defaultdict(float)
for i in tqdm(range(jobnum),title):
    frame.step()

    all_tasks_size[frame.step_host]+=sum(j.size for j in frame.step_j.tasks)
    r.append(sum(all_tasks_size.values())/(frame.num_of_cores*frame.num_of_hosts*max(max(i) for i in frame.cores_on_hosts)))
plt.title("utilization rate")
plt.plot(r,label=title)
plt.legend()

# GA
from stragety.GA import single_host,multi_hosts
title="GA stragety"
random.seed(seed),np.random.seed(seed)
frame=Frame(single_host,multi_hosts,lmd=lmd)
r=[]
all_tasks_size=defaultdict(float)
for i in tqdm(range(jobnum),title):
    frame.step()

    all_tasks_size[frame.step_host]+=sum(j.size for j in frame.step_j.tasks)
    r.append(sum(all_tasks_size.values())/(frame.num_of_cores*frame.num_of_hosts*max(max(i) for i in frame.cores_on_hosts)))

plt.plot(r,label=title)
plt.legend()



plt.show()