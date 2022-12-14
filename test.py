import random, numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from frame import Frame

seed=0
lmd=0.6
jobnum=20000

# main
from stragety import single_host,multi_hosts
title="main stragety"
random.seed(seed),np.random.seed(seed)
frame=Frame(single_host,multi_hosts,lmd=lmd)
y1,y2=[],[]
for i in tqdm(range(jobnum),title):
    frame.step()
    y1.append(frame.step_start_line-frame.step_j.arrive_time)
    y2.append(max(
        frame.cores_on_hosts[frame.step_host][c] 
        for c in frame.step_assignment
    )-frame.step_start_line)
plt.figure()
plt.title(title)
plt.plot(y1,label="wait time")
plt.plot(y2,label="execute time")
plt.legend()

# random
from stragety.Random import single_host,multi_hosts
title="random stragety"
random.seed(seed),np.random.seed(seed)
frame=Frame(single_host,multi_hosts,lmd=lmd)
y1,y2=[],[]
for i in tqdm(range(jobnum),title):
    frame.step()
    y1.append(frame.step_start_line-frame.step_j.arrive_time)
    y2.append(max(
        frame.cores_on_hosts[frame.step_host][c] 
        for c in frame.step_assignment
    )-frame.step_start_line)
plt.figure()
plt.title(title)
plt.plot(y1,label="wait time")
plt.plot(y2,label="execute time")
plt.legend()

# greedy
from stragety.Greedy import single_host,multi_hosts
title="greedy stragety"
random.seed(seed),np.random.seed(seed)
frame=Frame(single_host,multi_hosts,lmd=lmd)
y1,y2=[],[]
for i in tqdm(range(jobnum),title):
    frame.step()
    y1.append(frame.step_start_line-frame.step_j.arrive_time)
    y2.append(max(
        frame.cores_on_hosts[frame.step_host][c] 
        for c in frame.step_assignment
    )-frame.step_start_line)
plt.figure()
plt.title(title)
plt.plot(y1,label="wait time")
plt.plot(y2,label="execute time")
plt.legend()



plt.show()